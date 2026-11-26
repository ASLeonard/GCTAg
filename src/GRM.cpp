/*
   GCTA: a tool for Genome-wide Complex Trait Analysis

   New implementation: generate, read and process GRM.

   Depends on the class of genotype

   Developed by Zhili Zheng<zhilizheng@outlook.com>

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   A copy of the GNU General Public License is attached along with this program.
   If not, see <http://www.gnu.org/licenses/>.
*/

#include "GRM.h"
#include "GrmBedInt8.hpp"
#include "Logger.h"
#include "OptionIO.h"
#include "cpu.h"
#include <Eigen/Dense>
#include <format>
#include <limits>
#include <memory>
#include <iterator>
#include <algorithm>
#include <ranges>
#include <iomanip>
#include <sstream>
#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include "utils.hpp"
#include "grm_binary_io.hpp"
#include <omp.h>
#include "third_party/plink-ng/2.0/include/plink2_bits.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <csignal>

using std::to_string;

namespace gcta_grm_detail {

constexpr size_t kMaxPosixIoChunk = gcta_io::kMaxIoChunkBytes;
constexpr size_t kGrmAsyncSlotBytes = gcta_grm_io::kDefaultGrmIoSlotBytes; // 64 MiB

class GrmUniqueFd {
public:
    GrmUniqueFd() = default;
    explicit GrmUniqueFd(int fd) noexcept : fd_(fd) {}
    ~GrmUniqueFd() { reset(); }
    GrmUniqueFd(const GrmUniqueFd&) = delete;
    GrmUniqueFd& operator=(const GrmUniqueFd&) = delete;
    GrmUniqueFd(GrmUniqueFd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    GrmUniqueFd& operator=(GrmUniqueFd&& o) noexcept {
        if (this != &o) {
            reset(o.fd_);
            o.fd_ = -1;
        }
        return *this;
    }
    int get() const noexcept { return fd_; }
    int release() noexcept {
        int r = fd_;
        fd_ = -1;
        return r;
    }
    explicit operator bool() const noexcept { return fd_ >= 0; }
    void reset(int new_fd = -1) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = new_fd;
    }
private:
    int fd_ = -1;
};

inline GrmUniqueFd open_posix_read(const std::string& path) noexcept {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    return GrmUniqueFd(::open(path.c_str(), flags));
}

inline GrmUniqueFd open_posix_write_trunc(const std::string& path) noexcept {
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    return GrmUniqueFd(::open(path.c_str(), flags, 0644));
}

inline bool write_exact_posix(int fd, const void* buf, size_t count, int& err_out) noexcept {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        const size_t step = std::min(kMaxPosixIoChunk, count - done);
        const ssize_t w = ::write(fd, p + done, step);
        if (w < 0) {
            if (errno == EINTR) continue;
            err_out = errno;
            return false;
        }
        if (w == 0) {
            err_out = EIO;
            return false;
        }
        done += static_cast<size_t>(w);
    }
    return true;
}

inline std::pair<int, int> lower_tri_col_range(int cols, int tid, int nthreads) noexcept {
    if (cols <= 0 || nthreads <= 0) return {0, 0};
    if (nthreads == 1) return {0, cols};
    const double W = 0.5 * static_cast<double>(cols) * (static_cast<double>(cols) + 1.0);
    auto col_at = [&](int k) -> int {
        if (k <= 0) return 0;
        if (k >= nthreads) return cols;
        const double rem = W * (static_cast<double>(nthreads - k) / static_cast<double>(nthreads));
        const double tail = std::floor((-1.0 + std::sqrt(1.0 + 8.0 * rem)) * 0.5);
        return std::clamp(cols - static_cast<int>(tail), 0, cols);
    };
    return {col_at(tid), col_at(tid + 1)};
}

// Finds the largest `re` in (rs, full_re] such that a GRM tile spanning rows
// [rs, re) fits within budget_elems *actually touched* elements.
inline int solve_tile_budget_end(int rs, int full_re, double budget_elems) {
    const double drs = rs;
    const double disc = 1.0 + 4.0 * (2.0 * budget_elems + drs * (drs + 1.0));
    int re = static_cast<int>(std::floor((-1.0 + std::sqrt(disc)) / 2.0));
    return std::min(re, full_re);
}

// True when a single tile spanning the whole GRM [full_rs, full_re) already
// fits within budget_elems -- i.e. tiling would produce exactly one tile.
inline bool grm_tile_budget_covers_whole(int full_rs, int full_re, double budget_elems) {
    const double drs = full_rs, dre = full_re;
    const double whole_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
    return whole_elems <= budget_elems;
}

inline bool prefix_has_readable_grm(const std::string& pfx, bool require_n_for_f32) {
    if (!checkFileReadable(pfx + ".grm.id")) return false;
    if (checkFileReadable(pfx + ".grm.f16")) return true;
    if (checkFileReadable(pfx + ".grm.bin")) {
        return !require_n_for_f32 || checkFileReadable(pfx + ".grm.N.bin");
    }
    return false;
}

} // namespace gcta_grm_detail

// ---------------------------------------------------------------------------
// GRM reader-mode constructor (grm prune / grm sparse / grm merge / subtract / unify)
// ---------------------------------------------------------------------------
GRM::GRM(const std::string& grm_prefix, const std::string& keep, const std::string& remove,
         const std::string& out_prefix, bool detail, bool out_fp16)
    : write_fp16_(out_fp16), o_name(out_prefix), keep_file(keep), remove_file(remove), cutoff_detail(detail) {
    if (!grm_prefix.empty()) {
        this->grm_file = grm_prefix;
        const gcta_grm_io::ResolvedGrmInput resolved =
            gcta_grm_io::resolve_grm_input(grm_file, /*prefer_fp16=*/out_fp16);
        input_grm_format_    = resolved.format;
        input_grm_data_path_ = resolved.data_path;
        input_has_n_file_    = resolved.has_n_file;

        grm_ids = Pheno::read_sublist(resolved.id_path);
        num_subjects = grm_ids.size();
        const uint64_t expected_tri = num_subjects * (num_subjects + 1ULL) / 2ULL;

        const int fd = gcta_grm_io::open_readonly_checked(input_grm_data_path_);
        if (input_grm_format_ == gcta_grm_io::GrmDiskFormat::F16) {
            Eigen::VectorXf diag_vec;
            gcta_grm_io::read_grm_f16_header_and_diag(fd, input_grm_data_path_, grm_ids, input_f16_hdr_, diag_vec);
            ::close(fd);
            input_f16_diag_.assign(diag_vec.data(), diag_vec.data() + diag_vec.size());
        } else {
            struct stat st{};
            if (::fstat(fd, &st) != 0 || st.st_size < 0 ||
                static_cast<uint64_t>(st.st_size) != expected_tri * sizeof(float)) {
                ::close(fd);
                LOGGER.e(0, "The IDs in GRM and the IDs in the GRM binary file do not match [" + grm_file + "]");
            }
            ::close(fd);
        }

        index_keep.resize(num_subjects);
        std::iota(index_keep.begin(), index_keep.end(), 0);

        if (!keep_file.empty()) {
            vector<string> keep_lists = Pheno::read_sublist(keep_file);
            LOGGER << "Get " << keep_lists.size() << " samples from list [" << keep_file << "]." << std::endl;
            Pheno::set_keep(keep_lists, grm_ids, index_keep, true);
        }

        if (!remove_file.empty()) {
            vector<string> rm_lists = Pheno::read_sublist(remove_file);
            LOGGER << "Get " << rm_lists.size() << " samples from list [" << remove_file << "]." << std::endl;
            Pheno::set_keep(rm_lists, grm_ids, index_keep, false);
        }
    }
}

// ---------------------------------------------------------------------------
// GRM::subtract_grm — HPC-safe single-stream double-buffered LOCO subtraction
// Supports .grm.bin, .grm.f16 (zero-missing scalar N fast path), and mixed inputs.
// ---------------------------------------------------------------------------
void GRM::subtract_grm(string mgrm_file, string out_file, bool out_fp16){
    std::ifstream mgrm(mgrm_file.c_str());
    if (!mgrm) {
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }
    string line;
    vector<string> files;
    vector<string> err_files;
    while (getline(mgrm, line)) {
        boost::trim(line);
        if (!line.empty()) {
            if (gcta_grm_detail::prefix_has_readable_grm(line, /*require_n_for_f32=*/true)) {
                files.push_back(line);
            } else {
                err_files.push_back(line);
            }
        }
    }

    if (!err_files.empty()) {
        string out_err = "can't read GRM (*.grm.id, *.grm.bin/.grm.f16) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if (files.size() != 2) {
        LOGGER.e(0, "only 2 GRMs are supported currently.");
    }

    const std::string id0_path = files[0] + ".grm.id";
    const std::string id1_path = files[1] + ".grm.id";
    LOGGER.i(0, "Reading [" + id0_path + "]...");
    vector<string> common_id = Pheno::read_sublist(id0_path);
    LOGGER << common_id.size() << " samples have been read." << std::endl;
    if (!gcta_grm_io::files_byte_identical(id0_path, id1_path)) {
        LOGGER.i(0, "Reading [" + id1_path + "]...");
        vector<string> common_id2 = Pheno::read_sublist(id1_path);
        if (common_id != common_id2) {
            LOGGER.e(0, "The sample IDs in the two GRMs are not same. You may try 'grm unify' first.");
        }
    }
    gcta_grm_io::copy_file_posix(id0_path, out_file + ".grm.id");

    const uint64_t n_sub = static_cast<uint64_t>(common_id.size());
    const uint64_t grm_size = (1ULL + n_sub) * n_sub / 2ULL;
    const uint64_t grm_f32_size = grm_size * sizeof(float);

    struct SubInput {
        gcta_grm_detail::GrmUniqueFd val_fd;
        gcta_grm_detail::GrmUniqueFd n_fd;
        std::string val_path;
        std::string n_path;
        bool is_fp16 = false;
        bool zero_missing = false;
        off_t body_off = 0;
        double const_n = 0.0;
        uint64_t valid_markers = 0;
        Eigen::VectorXf diag_f32;
        gcta_grm_io::GrmFp16Header hdr{};
    };

    std::array<SubInput, 2> in;
    for (int f = 0; f < 2; ++f) {
        const auto res = gcta_grm_io::resolve_grm_input(files[static_cast<size_t>(f)], out_fp16);
        in[static_cast<size_t>(f)].val_path = res.data_path;
        in[static_cast<size_t>(f)].n_path   = res.n_path;
        in[static_cast<size_t>(f)].is_fp16  = (res.format == gcta_grm_io::GrmDiskFormat::F16);
        in[static_cast<size_t>(f)].val_fd   = gcta_grm_detail::open_posix_read(res.data_path);
        if (!in[static_cast<size_t>(f)].val_fd) {
            LOGGER.e(0, "can't open [" + res.data_path + "].");
        }
        if (in[static_cast<size_t>(f)].is_fp16) {
            gcta_grm_io::read_grm_f16_header_and_diag(
                in[static_cast<size_t>(f)].val_fd.get(), res.data_path, common_id,
                in[static_cast<size_t>(f)].hdr, in[static_cast<size_t>(f)].diag_f32);
            in[static_cast<size_t>(f)].body_off = static_cast<off_t>(in[static_cast<size_t>(f)].hdr.body_offset);
            in[static_cast<size_t>(f)].zero_missing =
                ((in[static_cast<size_t>(f)].hdr.flags & gcta_grm_io::kGrmF16FlagZeroMissing) != 0);
            in[static_cast<size_t>(f)].valid_markers = in[static_cast<size_t>(f)].hdr.valid_markers;
            if (in[static_cast<size_t>(f)].zero_missing) {
                in[static_cast<size_t>(f)].const_n = (in[static_cast<size_t>(f)].hdr.valid_markers > 0)
                    ? static_cast<double>(in[static_cast<size_t>(f)].hdr.valid_markers)
                    : in[static_cast<size_t>(f)].hdr.m_snps_mean;
            } else {
                in[static_cast<size_t>(f)].n_fd = gcta_grm_detail::open_posix_read(res.n_path);
                struct stat st{};
                if (!in[static_cast<size_t>(f)].n_fd || ::fstat(in[static_cast<size_t>(f)].n_fd.get(), &st) != 0 ||
                    st.st_size < 0 || static_cast<uint64_t>(st.st_size) != grm_f32_size) {
                    LOGGER.e(0, "The size of [" + res.n_path + "] is not correct.");
                }
            }
        } else {
            struct stat st{};
            if (::fstat(in[static_cast<size_t>(f)].val_fd.get(), &st) != 0 ||
                st.st_size < 0 || static_cast<uint64_t>(st.st_size) != grm_f32_size) {
                LOGGER.e(0, "The size of [" + res.data_path + "] is not correct.");
            }
            in[static_cast<size_t>(f)].n_fd = gcta_grm_detail::open_posix_read(res.n_path);
            if (!in[static_cast<size_t>(f)].n_fd || ::fstat(in[static_cast<size_t>(f)].n_fd.get(), &st) != 0 ||
                st.st_size < 0 || static_cast<uint64_t>(st.st_size) != grm_f32_size) {
                LOGGER.e(0, "The size of [" + res.n_path + "] is not correct.");
            }
        }
    }

    const bool write_fp16 = out_fp16 || (in[0].is_fp16 && in[1].is_fp16);
    const bool both_zero_missing = in[0].zero_missing && in[1].zero_missing;
    const bool write_n_file = !(write_fp16 && both_zero_missing);

    if (both_zero_missing && in[0].const_n < in[1].const_n) {
        LOGGER.e(0, "subtract_grm: the first GRM [" + files[0] + "] has fewer markers than "
                    "the second [" + files[1] + "]. The larger (all-chromosome) GRM must be listed first.");
    }

    constexpr uint64_t kChunkElems = 16ULL << 20; // 16M elems (64 MiB f32 / 32 MiB f16)
    const size_t chunk_elems = static_cast<size_t>(std::max<uint64_t>(1, std::min(grm_size, kChunkElems)));

    LOGGER.i(0, "Subtracting GRMs (single-stream async, output=" +
                std::string(write_fp16 ? ".grm.f16" : ".grm.bin") + ")...");

    const std::string po_grm  = out_file + (write_fp16 ? ".grm.f16" : ".grm.bin");
    const std::string po_grmN = out_file + ".grm.N.bin";
    const uint64_t out_body_off = write_fp16 ? gcta_grm_io::grm_f16_body_offset(n_sub) : 0ULL;

    auto fdo_grm = gcta_grm_detail::open_posix_write_trunc(po_grm);
    if (!fdo_grm) LOGGER.error(0, "can't write to [{}].", po_grm);
    gcta_grm_detail::GrmUniqueFd fdo_grmN;
    if (write_n_file) {
        fdo_grmN = gcta_grm_detail::open_posix_write_trunc(po_grmN);
        if (!fdo_grmN) LOGGER.error(0, "can't write to [{}].", po_grmN);
    }

    if (write_fp16) {
        gcta_grm_io::GrmFp16Header draft{};
        std::memcpy(draft.magic, gcta_grm_io::kGrmF16MagicDraft, sizeof(draft.magic));
        draft.version = gcta_grm_io::kGrmF16Version;
        draft.dtype = gcta_grm_io::kGrmF16DtypeFp16;
        draft.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
        draft.n = n_sub;
        draft.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
        draft.body_offset = out_body_off;
        draft.part_row_start = 0;
        draft.part_row_end = n_sub;
        std::vector<char> prefix(static_cast<size_t>(out_body_off), 0);
        std::memcpy(prefix.data(), &draft, sizeof(draft));
        try {
            gcta_io::write_exact_throwing(fdo_grm.get(), prefix.data(), prefix.size(), po_grm);
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("subtract_grm: ") + e.what());
        }
    }

    // Double-buffered 1-thread sequential reader
    struct SubInSlot {
        AlignedRawBuffer<char, 64> g1, g2;
        AlignedRawBuffer<float, 64> n1, n2;
        size_t elems = 0;
        size_t offset = 0;
    };
    std::array<SubInSlot, 2> rslots;
    for (int s = 0; s < 2; ++s) {
        rslots[static_cast<size_t>(s)].g1.resize(chunk_elems * (in[0].is_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float)));
        rslots[static_cast<size_t>(s)].g2.resize(chunk_elems * (in[1].is_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float)));
        if (!in[0].zero_missing) rslots[static_cast<size_t>(s)].n1.resize(chunk_elems);
        if (!in[1].zero_missing) rslots[static_cast<size_t>(s)].n2.resize(chunk_elems);
    }

    std::mutex rmu;
    std::condition_variable_any r_ready_cv, r_free_cv;
    std::array<bool, 2> r_busy{false, false};
    std::array<int, 2> r_ring{0, 0};
    int r_head = 0, r_tail = 0, r_count = 0;
    bool r_failed = false;
    std::exception_ptr r_err;

    std::jthread reader([&](std::stop_token st) {
        try {
            size_t seq = 0;
            for (size_t offset = 0; offset < grm_size && !st.stop_requested(); offset += chunk_elems) {
                const size_t elems = static_cast<size_t>(std::min<uint64_t>(chunk_elems, grm_size - offset));
                const int slot = static_cast<int>(seq & 1u);
                ++seq;
                {
                    std::unique_lock<std::mutex> lock(rmu);
                    r_free_cv.wait(lock, st, [&]() { return !r_busy[static_cast<size_t>(slot)]; });
                    if (st.stop_requested()) return;
                    r_busy[static_cast<size_t>(slot)] = true;
                }
                auto& rs = rslots[static_cast<size_t>(slot)];
                const size_t b0 = elems * (in[0].is_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float));
                const size_t b1 = elems * (in[1].is_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float));
                gcta_io::read_exact_at_throwing(in[0].val_fd.get(), rs.g1.data(), b0,
                    in[0].body_off + static_cast<off_t>(offset * (in[0].is_fp16 ? 2u : 4u)), in[0].val_path);
                if (!in[0].zero_missing) {
                    gcta_io::read_exact_at_throwing(in[0].n_fd.get(), rs.n1.data(), elems * sizeof(float),
                        static_cast<off_t>(offset * sizeof(float)), in[0].n_path);
                }
                gcta_io::read_exact_at_throwing(in[1].val_fd.get(), rs.g2.data(), b1,
                    in[1].body_off + static_cast<off_t>(offset * (in[1].is_fp16 ? 2u : 4u)), in[1].val_path);
                if (!in[1].zero_missing) {
                    gcta_io::read_exact_at_throwing(in[1].n_fd.get(), rs.n2.data(), elems * sizeof(float),
                        static_cast<off_t>(offset * sizeof(float)), in[1].n_path);
                }
                {
                    std::lock_guard<std::mutex> lock(rmu);
                    rs.elems = elems;
                    rs.offset = offset;
                    r_ring[static_cast<size_t>(r_tail)] = slot;
                    r_tail = (r_tail + 1) & 1;
                    ++r_count;
                }
                r_ready_cv.notify_one();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(rmu);
                r_err = std::current_exception();
                r_failed = true;
            }
            r_ready_cv.notify_all();
        }
    });

    // Double-buffered 1-thread sequential writer (writes grm then N sequentially)
    struct SubOutSlot {
        AlignedRawBuffer<char, 64> g;
        AlignedRawBuffer<float, 64> n;
        size_t g_bytes = 0;
        size_t n_bytes = 0;
    };
    const size_t out_g_bpe = write_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    std::array<SubOutSlot, 2> wslots;
    for (int s = 0; s < 2; ++s) {
        wslots[static_cast<size_t>(s)].g.resize(chunk_elems * out_g_bpe);
        if (write_n_file) wslots[static_cast<size_t>(s)].n.resize(chunk_elems);
    }

    std::mutex wmu;
    std::condition_variable_any w_work_cv, w_space_cv;
    std::array<bool, 2> w_queued{false, false};
    std::array<int, 2> w_ring{0, 0};
    int w_head = 0, w_tail = 0, w_count = 0;
    bool w_closing = false, w_abort = false, w_failed = false;
    std::exception_ptr w_err;

    std::jthread writer([&](std::stop_token st) {
        while (true) {
            int slot = -1;
            {
                std::unique_lock<std::mutex> lock(wmu);
                w_work_cv.wait(lock, st, [&]() { return w_count > 0 || w_closing || w_abort || w_failed; });
                if (w_abort || w_failed) return;
                if (w_count == 0) {
                    if (w_closing || st.stop_requested()) return;
                    continue;
                }
                slot = w_ring[static_cast<size_t>(w_head)];
                w_head = (w_head + 1) & 1;
                --w_count;
            }
            try {
                const auto& ws = wslots[static_cast<size_t>(slot)];
                if (ws.g_bytes > 0) gcta_io::write_exact_throwing(fdo_grm.get(), ws.g.data(), ws.g_bytes, po_grm);
                if (write_n_file && ws.n_bytes > 0) gcta_io::write_exact_throwing(fdo_grmN.get(), ws.n.data(), ws.n_bytes, po_grmN);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(wmu);
                    w_err = std::current_exception();
                    w_failed = true;
                    w_queued[static_cast<size_t>(slot)] = false;
                }
                w_space_cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(wmu);
                w_queued[static_cast<size_t>(slot)] = false;
            }
            w_space_cv.notify_all();
        }
    });

    std::vector<float> out_diag_f32;
    if (write_fp16) out_diag_f32.assign(static_cast<size_t>(n_sub), 0.0f);
    double tr_K = 0.0, tr_K2 = 0.0, diag_n_sum = 0.0;
    uint64_t next_diag_row = 0;
    std::vector<std::pair<uint64_t, size_t>> chunk_diags;
    int next_wslot = 0;

    auto abort_threads = [&]() {
        reader.request_stop();
        r_free_cv.notify_all();
        if (reader.joinable()) reader.join();
        {
            std::lock_guard<std::mutex> lock(wmu);
            w_abort = true;
            w_closing = true;
        }
        writer.request_stop();
        w_work_cv.notify_all();
        if (writer.joinable()) writer.join();
    };

    try {
        for (size_t offset = 0; offset < grm_size; offset += chunk_elems) {
            int rslot = -1;
            {
                std::unique_lock<std::mutex> lock(rmu);
                r_ready_cv.wait(lock, [&]() { return r_failed || r_count > 0; });
                if (r_failed) std::rethrow_exception(r_err);
                rslot = r_ring[static_cast<size_t>(r_head)];
                r_head = (r_head + 1) & 1;
                --r_count;
            }
            const auto& rs = rslots[static_cast<size_t>(rslot)];
            const size_t elems = rs.elems;

            if (offset == 0 && elems > 0) {
                const double n1_0 = in[0].zero_missing ? in[0].const_n : static_cast<double>(rs.n1[0]);
                const double n2_0 = in[1].zero_missing ? in[1].const_n : static_cast<double>(rs.n2[0]);
                if (n1_0 < n2_0) {
                    throw std::runtime_error(
                        "subtract_grm: the first GRM [" + files[0] + "] has fewer markers than "
                        "the second [" + files[1] + "] in the first pair. "
                        "The larger (all-chromosome) GRM must be listed first.");
                }
            }

            chunk_diags.clear();
            while (next_diag_row < n_sub) {
                const uint64_t dpos = next_diag_row * (next_diag_row + 1ULL) / 2ULL + next_diag_row;
                if (dpos >= offset + elems) break;
                chunk_diags.emplace_back(next_diag_row, static_cast<size_t>(dpos - offset));
                ++next_diag_row;
            }

            int wslot = -1;
            {
                std::unique_lock<std::mutex> lock(wmu);
                w_space_cv.wait(lock, [&]() { return w_failed || !w_queued[0] || !w_queued[1]; });
                if (w_failed) std::rethrow_exception(w_err);
                wslot = !w_queued[static_cast<size_t>(next_wslot)] ? next_wslot : (1 - next_wslot);
                next_wslot = 1 - wslot;
            }
            auto& ws = wslots[static_cast<size_t>(wslot)];

            const bool g1_f16 = in[0].is_fp16;
            const bool g2_f16 = in[1].is_fp16;
            const bool n1_const = in[0].zero_missing;
            const bool n2_const = in[1].zero_missing;
            const double cn1 = in[0].const_n;
            const double cn2 = in[1].const_n;

            const auto* p1_f16 = g1_f16 ? reinterpret_cast<const gcta_grm_io::GrmFp16*>(rs.g1.data()) : nullptr;
            const auto* p1_f32 = g1_f16 ? nullptr : reinterpret_cast<const float*>(rs.g1.data());
            const auto* p2_f16 = g2_f16 ? reinterpret_cast<const gcta_grm_io::GrmFp16*>(rs.g2.data()) : nullptr;
            const auto* p2_f32 = g2_f16 ? nullptr : reinterpret_cast<const float*>(rs.g2.data());
            const float* pn1 = n1_const ? nullptr : rs.n1.data();
            const float* pn2 = n2_const ? nullptr : rs.n2.data();

            if (write_fp16) {
                auto* og = reinterpret_cast<gcta_grm_io::GrmFp16*>(ws.g.data());
                float* on = write_n_file ? ws.n.data() : nullptr;
                double chunk_all_sq = 0.0;
                #pragma omp parallel for reduction(+:chunk_all_sq) schedule(static)
                for (size_t j = 0; j < elems; ++j) {
                    const double v1 = g1_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p1_f16[j]))
                                             : static_cast<double>(p1_f32[j]);
                    const double v2 = g2_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p2_f16[j]))
                                             : static_cast<double>(p2_f32[j]);
                    const double nn1 = n1_const ? cn1 : static_cast<double>(pn1[j]);
                    const double nn2 = n2_const ? cn2 : static_cast<double>(pn2[j]);
                    const double nd = nn1 - nn2;
                    const float res = (nd > 0.0) ? static_cast<float>((v1 * nn1 - v2 * nn2) / nd) : 0.0f;
                    og[j] = gcta_grm_io::f32_to_fp16(res);
                    if (on) on[j] = static_cast<float>(nd);
                    const double rd = static_cast<double>(res);
                    chunk_all_sq += rd * rd;
                }
                double diag_sq_old = 0.0;
                double diag_sq_new = 0.0;
                for (const auto& [r, loc] : chunk_diags) {
                    const double v1_q = g1_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p1_f16[loc]))
                                               : static_cast<double>(p1_f32[loc]);
                    const double v2_q = g2_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p2_f16[loc]))
                                               : static_cast<double>(p2_f32[loc]);
                    const double v1_e = g1_f16 ? static_cast<double>(in[0].diag_f32[static_cast<Eigen::Index>(r)]) : v1_q;
                    const double v2_e = g2_f16 ? static_cast<double>(in[1].diag_f32[static_cast<Eigen::Index>(r)]) : v2_q;
                    const double nn1 = n1_const ? cn1 : static_cast<double>(pn1[loc]);
                    const double nn2 = n2_const ? cn2 : static_cast<double>(pn2[loc]);
                    const double nd = nn1 - nn2;
                    const float old_d = (nd > 0.0) ? static_cast<float>((v1_q * nn1 - v2_q * nn2) / nd) : 0.0f;
                    const float exact_d = (nd > 0.0) ? static_cast<float>((v1_e * nn1 - v2_e * nn2) / nd) : 0.0f;
                    og[loc] = gcta_grm_io::f32_to_fp16(exact_d);
                    out_diag_f32[static_cast<size_t>(r)] = exact_d;
                    diag_n_sum += nd;
                    tr_K += static_cast<double>(exact_d);
                    diag_sq_old += static_cast<double>(old_d) * static_cast<double>(old_d);
                    diag_sq_new += static_cast<double>(exact_d) * static_cast<double>(exact_d);
                }
                tr_K2 += 2.0 * (chunk_all_sq - diag_sq_old) + diag_sq_new;
                ws.g_bytes = elems * sizeof(gcta_grm_io::GrmFp16);
                ws.n_bytes = write_n_file ? (elems * sizeof(float)) : 0;
            } else {
                auto* og = reinterpret_cast<float*>(ws.g.data());
                float* on = ws.n.data();
                #pragma omp parallel for schedule(static)
                for (size_t j = 0; j < elems; ++j) {
                    const double v1 = g1_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p1_f16[j]))
                                             : static_cast<double>(p1_f32[j]);
                    const double v2 = g2_f16 ? static_cast<double>(gcta_grm_io::fp16_to_f32(p2_f16[j]))
                                             : static_cast<double>(p2_f32[j]);
                    const float fn1 = n1_const ? static_cast<float>(cn1) : pn1[j];
                    const float fn2 = n2_const ? static_cast<float>(cn2) : pn2[j];
                    const float fnd = fn1 - fn2;
                    on[j] = fnd;
                    og[j] = (fnd > 0.0f)
                        ? static_cast<float>((v1 * static_cast<double>(fn1) - v2 * static_cast<double>(fn2)) /
                                             static_cast<double>(fnd))
                        : 0.0f;
                }
                for (const auto& [r, loc] : chunk_diags) {
                    if (g1_f16 || g2_f16) {
                        const double v1 = g1_f16 ? static_cast<double>(in[0].diag_f32[static_cast<Eigen::Index>(r)])
                                                 : static_cast<double>(p1_f32[loc]);
                        const double v2 = g2_f16 ? static_cast<double>(in[1].diag_f32[static_cast<Eigen::Index>(r)])
                                                 : static_cast<double>(p2_f32[loc]);
                        const float fn1 = n1_const ? static_cast<float>(cn1) : pn1[loc];
                        const float fn2 = n2_const ? static_cast<float>(cn2) : pn2[loc];
                        const float fnd = fn1 - fn2;
                        og[loc] = (fnd > 0.0f)
                            ? static_cast<float>((v1 * static_cast<double>(fn1) - v2 * static_cast<double>(fn2)) /
                                                 static_cast<double>(fnd))
                            : 0.0f;
                    }
                }
                ws.g_bytes = elems * sizeof(float);
                ws.n_bytes = elems * sizeof(float);
            }

            {
                std::lock_guard<std::mutex> lock(rmu);
                r_busy[static_cast<size_t>(rslot)] = false;
            }
            r_free_cv.notify_one();

            {
                std::lock_guard<std::mutex> lock(wmu);
                if (w_failed) std::rethrow_exception(w_err);
                w_queued[static_cast<size_t>(wslot)] = true;
                w_ring[static_cast<size_t>(w_tail)] = wslot;
                w_tail = (w_tail + 1) & 1;
                ++w_count;
            }
            w_work_cv.notify_one();
        }

        if (reader.joinable()) reader.join();
        if (r_failed) std::rethrow_exception(r_err);

        {
            std::lock_guard<std::mutex> lock(wmu);
            w_closing = true;
        }
        w_work_cv.notify_all();
        if (writer.joinable()) writer.join();
        if (w_failed) std::rethrow_exception(w_err);

        if (write_fp16) {
            gcta_grm_io::GrmFp16Header final_hdr{};
            std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
            final_hdr.version = gcta_grm_io::kGrmF16Version;
            final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
            final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized | gcta_grm_io::kGrmF16FlagHasTraces |
                              gcta_grm_io::kGrmF16FlagHasDiagF32 |
                              (both_zero_missing ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u);
            final_hdr.n = n_sub;
            final_hdr.m_snps_mean = both_zero_missing
                ? std::max(0.0, in[0].const_n - in[1].const_n)
                : (n_sub > 0 ? diag_n_sum / static_cast<double>(n_sub) : 0.0);
            const uint64_t diff_vm = (in[0].valid_markers >= in[1].valid_markers)
                ? (in[0].valid_markers - in[1].valid_markers)
                : 0ULL;
            final_hdr.valid_markers = (diff_vm > 0)
                ? diff_vm
                : (final_hdr.m_snps_mean > 0.0
                       ? static_cast<uint64_t>(std::llround(final_hdr.m_snps_mean))
                       : 0ULL);
            final_hdr.id_hash = gcta_io::hash_sample_ids(common_id);
            final_hdr.trace_K = tr_K;
            final_hdr.trace_K2 = tr_K2;
            final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.body_offset = out_body_off;
            final_hdr.part_row_start = 0;
            final_hdr.part_row_end = n_sub;
            final_hdr.mtd_weight = 1.0;

            if (n_sub > 0) {
                gcta_io::pwrite_exact_throwing(
                    fdo_grm.get(), out_diag_f32.data(),
                    static_cast<size_t>(n_sub) * sizeof(float),
                    static_cast<off_t>(final_hdr.diag_offset), po_grm);
            }
            gcta_io::pwrite_exact_throwing(fdo_grm.get(), &final_hdr, sizeof(final_hdr), 0, po_grm);
        }
    } catch (const std::exception& e) {
        abort_threads();
        LOGGER.e(0, std::string("subtract_grm: ") + e.what());
    }

    fdo_grm.reset();
    fdo_grmN.reset();
    LOGGER.i(0, "The subtracted GRM has been written to [" + po_grm +
                (write_n_file ? ", " + po_grmN : " (zero-missing; .grm.N.bin omitted)") + "].");
}

void GRM::merge_grm(string mgrm_file, string out_file, int memory_budget_gb, bool out_fp16){
    std::ifstream mgrm(mgrm_file.c_str());
    if (!mgrm) {
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }

    string line;
    vector<string> files;
    vector<string> err_files;
    while (getline(mgrm, line)) {
        boost::trim(line);
        if (!line.empty()) {
            if (gcta_grm_detail::prefix_has_readable_grm(line, /*require_n_for_f32=*/true)) {
                files.push_back(line);
            } else {
                err_files.push_back(line);
            }
        }
    }

    if (!err_files.empty()) {
        string out_err = "can't read GRM (*.grm.id, *.grm.bin/.grm.f16, *.grm.N.bin) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if (files.size() < 2) {
        LOGGER.e(0, "streaming merge needs at least two GRMs in --mgrm.");
    }

    gcta_grm_io::merge_grms(files, out_file, memory_budget_gb, out_fp16);
}

// ---------------------------------------------------------------------------
// GRM::unify_grm — HPC-safe sequential unification (zero multi-file OMP storm,
// zero O(N^2) 4-byte fseeko+fread seeks, zero std::endl flushes, .grm.f16 aware).
// ---------------------------------------------------------------------------
void GRM::unify_grm(string mgrm_file, string out_file, bool out_fp16){
    std::ifstream mgrm(mgrm_file.c_str());
    if (!mgrm) {
        LOGGER.e(0, "can't open " + mgrm_file + " to read.");
    }
    string line;
    vector<string> files;
    vector<string> err_files;
    while (getline(mgrm, line)) {
        boost::trim(line);
        if (!line.empty()) {
            if (gcta_grm_detail::prefix_has_readable_grm(line, /*require_n_for_f32=*/false)) {
                files.push_back(line);
            } else {
                err_files.push_back(line);
            }
        }
    }

    if (!err_files.empty()) {
        string out_err = "can't read GRM (*.grm.id, *.grm.bin/.grm.f16) in ";
        out_err += boost::algorithm::join(err_files, ", ");
        out_err += ".";
        LOGGER.e(0, out_err);
    }
    if (files.size() < 2) {
        LOGGER.e(0, "not enough valid GRMs to be unified.");
    }

    LOGGER.i(0, "Reading [" + files[0] + ".grm.id]...");
    vector<string> common_id = Pheno::read_sublist(files[0] + ".grm.id");
    LOGGER << common_id.size() << " samples have been read." << std::endl;
    vector<vector<string>> ids(files.size());
    ids[0] = common_id;
    for (size_t i = 1; i < files.size(); i++) {
        string cur_file = files[i] + ".grm.id";
        LOGGER.i(0, "Reading [" + cur_file + "]...");
        ids[i] = Pheno::read_sublist(cur_file);
        LOGGER << ids[i].size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, ids[i], index1, index2);
        vector<string> temp_common_id(index1.size());
        std::ranges::transform(index1, temp_common_id.begin(), [&common_id](uint32_t v) {
            return common_id[v];
        });
        common_id = std::move(temp_common_id);
        LOGGER << common_id.size() << " common samples in GRMs" << std::endl;
    }
    if (!keep_file.empty()) {
        LOGGER.i(0, "Keeping individuals listed in [" + keep_file + "]...");
        vector<string> keep_id = Pheno::read_sublist(keep_file);
        LOGGER << keep_id.size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, keep_id, index1, index2);
        vector<string> temp_common_id(index1.size());
        std::ranges::transform(index1, temp_common_id.begin(), [&common_id](uint32_t v) {
            return common_id[v];
        });
        common_id = std::move(temp_common_id);
        LOGGER << common_id.size() << " common samples after merging." << std::endl;
    }
    if (!remove_file.empty()) {
        LOGGER.i(0, "Excluding individuals listed in [" + remove_file + "]...");
        vector<string> remove_id = Pheno::read_sublist(remove_file);
        LOGGER << remove_id.size() << " samples have been read." << std::endl;
        vector<uint32_t> index1, index2;
        vector_commonIndex_sorted1(common_id, remove_id, index1, index2);
        vector<uint32_t> keeps(common_id.size());
        std::iota(keeps.begin(), keeps.end(), 0);
        vector<uint32_t> remain_index;
        std::set_difference(keeps.begin(), keeps.end(), index1.begin(), index1.end(), std::back_inserter(remain_index));
        vector<string> remain_ids(remain_index.size());
        std::ranges::transform(remain_index, remain_ids.begin(), [&common_id](uint32_t v) {
            return common_id[v];
        });
        common_id = std::move(remain_ids);
        LOGGER << common_id.size() << " common samples after merging." << std::endl;
    }

    vector<vector<uint32_t>> grm_indices(files.size());
    vector<string> output_fileNames;
    const string& out_string = out_file;
    const auto basename_out_result = getFileName(out_string);
    if (!basename_out_result) {
        LOGGER.e(0, "Invalid output path [" + out_string + "]: " + std::string(getUtilsErrorMessage(basename_out_result.error())));
    }
    const auto path_out_result = getPathName(out_string);
    if (!path_out_result) {
        LOGGER.e(0, "Invalid output path [" + out_string + "]: " + std::string(getUtilsErrorMessage(path_out_result.error())));
    }
    const string basename_out = basename_out_result.value();
    const string path_out = path_out_result.value();
    for (auto& filename : files) {
        const auto basename_result = getFileName(filename);
        if (!basename_result) {
            LOGGER.e(0, "Invalid GRM file path [" + filename + "]: " + std::string(getUtilsErrorMessage(basename_result.error())));
        }
        const auto output_path_result = joinPath(path_out, basename_result.value() + "_" + basename_out);
        if (!output_path_result) {
            LOGGER.e(0, "Failed to compose output path for [" + filename + "]: " + std::string(getUtilsErrorMessage(output_path_result.error())));
        }
        output_fileNames.push_back(output_path_result.value());
    }

    // Compute per-file mappings and write .grm.id sequentially (zero std::endl).
    for (size_t i = 0; i < files.size(); i++) {
        string id_file_name = output_fileNames[i] + ".grm.id";
        vector<uint32_t> index1;
        vector_commonIndex_sorted1(common_id, ids[i], index1, grm_indices[i]);
        vector<string> out_ids;
        out_ids.reserve(grm_indices[i].size());
        for (uint32_t idx : grm_indices[i]) {
            out_ids.push_back(ids[i][idx]);
        }
        gcta_grm_io::write_id_file_buffered(id_file_name, out_ids);
        LOGGER.i(0, "Unified individual IDs have been saved to [" + id_file_name + "].");
    }

    LOGGER.i(0, "Writing unified GRMs sequentially...");
    const uint64_t common_id_hash = gcta_io::hash_sample_ids(common_id);

    // Process one GRM at a time (strictly <= 1 sequential reader + <= 1 sequential writer).
    for (size_t i = 0; i < grm_indices.size(); i++) {
        const vector<uint32_t>& p_index = grm_indices[i];
        const int size_grm = static_cast<int>(p_index.size());
        const int largest_grm_size = static_cast<int>(ids[i].size());

        const auto resolved = gcta_grm_io::resolve_grm_input(files[i], out_fp16);
        const bool in_is_f16 = (resolved.format == gcta_grm_io::GrmDiskFormat::F16);
        const bool emit_f16 = out_fp16 || in_is_f16;
        const size_t in_bpe = in_is_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
        const size_t out_bpe = emit_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);

        const int in_fd = gcta_grm_io::open_readonly_checked(resolved.data_path);
        gcta_grm_io::GrmFp16Header in_hdr{};
        Eigen::VectorXf in_diag;
        off_t in_body_off = 0;
        if (in_is_f16) {
            gcta_grm_io::read_grm_f16_header_and_diag(in_fd, resolved.data_path, ids[i], in_hdr, in_diag);
            in_body_off = static_cast<off_t>(in_hdr.body_offset);
        } else {
            gcta_grm_io::detail::validate_f32_bin_size(in_fd, resolved.data_path, largest_grm_size);
        }

        const string wfile_name = output_fileNames[i] + (emit_f16 ? ".grm.f16" : ".grm.bin");
        const uint64_t out_body_off = emit_f16 ? gcta_grm_io::grm_f16_body_offset(static_cast<uint64_t>(size_grm)) : 0ULL;

        std::vector<int> kp(static_cast<size_t>(size_grm));
        bool is_monotonic = true;
        for (int j = 0; j < size_grm; ++j) {
            kp[static_cast<size_t>(j)] = static_cast<int>(p_index[static_cast<size_t>(j)]);
            if (j > 0 && kp[static_cast<size_t>(j)] <= kp[static_cast<size_t>(j - 1)]) {
                is_monotonic = false;
            }
        }

        std::vector<float> out_diag;
        if (emit_f16) out_diag.assign(static_cast<size_t>(size_grm), 0.0f);
        double tr_K = 0.0, tr_K2 = 0.0;

        try {
            gcta_io::AsyncSeqWriter writer(wfile_name, gcta_grm_detail::kGrmAsyncSlotBytes,
                                           static_cast<off_t>(out_body_off));
            if (emit_f16) {
                gcta_grm_io::write_grm_f16_draft_prefix(writer, static_cast<uint64_t>(size_grm));
            }

            if (is_monotonic && size_grm > 0) {
                const bool is_identity = (size_grm == largest_grm_size && kp.front() == 0 && kp.back() == size_grm - 1);
                std::vector<gcta_grm_io::KpRun> kp_runs;
                if (!is_identity) {
                    auto runs = gcta_grm_io::build_kp_runs(kp);
                    if (runs.size() * 2 < static_cast<size_t>(size_grm)) {
                        kp_runs = std::move(runs);
                    }
                }

                const size_t max_in_elems = std::max<size_t>(1, gcta_grm_detail::kGrmAsyncSlotBytes / in_bpe);
                std::vector<gcta_grm_io::detail::RowChunkPlan> plans;
                std::vector<std::pair<off_t, size_t>> io_chunks;
                gcta_grm_io::detail::build_row_chunk_plan(
                    largest_grm_size, &kp, is_identity, max_in_elems,
                    in_body_off, in_bpe, plans, io_chunks);

                gcta_io::AsyncSeqReader reader(in_fd, resolved.data_path, std::move(io_chunks),
                                               /*max_slot_bytes=*/0, /*take_ownership=*/false);
                const char* raw_ptr = nullptr;
                size_t raw_len = 0;
                size_t pidx = 0;
                while (reader.next_slot(raw_ptr, raw_len)) {
                    const auto& plan = plans[pidx++];
                    const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
                    const int j_lo = plan.j_lo;
                    const int j_hi = plan.j_hi;
                    const size_t out_base = static_cast<size_t>(j_lo) * (j_lo + 1) / 2;
                    const size_t out_end  = static_cast<size_t>(j_hi) * (j_hi + 1) / 2;
                    const size_t out_elems = out_end - out_base;
                    const size_t out_bytes = out_elems * out_bpe;

                    AlignedRawBuffer<char, 64>& wslot = writer.acquire_slot();
                    if (wslot.size() < out_bytes) wslot.resize(out_bytes);

                    #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                    for (int j = j_lo; j < j_hi; ++j) {
                        const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                        const size_t src_off = static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base;
                        const size_t dst_off = static_cast<size_t>(j) * (j + 1) / 2 - out_base;
                        double row_off_sq = 0.0;

                        auto read_elem = [&](int c_in) -> float {
                            if (in_is_f16) {
                                const auto* p = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + src_off;
                                return gcta_grm_io::fp16_to_f32(p[c_in]);
                            }
                            const auto* p = reinterpret_cast<const float*>(raw_ptr) + src_off;
                            return p[c_in];
                        };

                        const float diag_v = in_is_f16 ? in_diag[r_in] : read_elem(r_in);
                        if (emit_f16) {
                            auto* dst = reinterpret_cast<gcta_grm_io::GrmFp16*>(wslot.data()) + dst_off;
                            if (is_identity && in_is_f16) {
                                const auto* src = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + src_off;
                                std::memcpy(dst, src, static_cast<size_t>(j + 1) * sizeof(gcta_grm_io::GrmFp16));
                                #pragma omp simd reduction(+:row_off_sq)
                                for (int c = 0; c < j; ++c) {
                                    const double vd = static_cast<double>(gcta_grm_io::fp16_to_f32(src[c]));
                                    row_off_sq += vd * vd;
                                }
                            } else if (!kp_runs.empty()) {
                                for (const auto& run : kp_runs) {
                                    if (run.dst_start >= j) break;
                                    const int cnt = std::min(run.len, j - run.dst_start);
                                    for (int k = 0; k < cnt; ++k) {
                                        const float v = read_elem(run.src_start + k);
                                        dst[run.dst_start + k] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        row_off_sq += vd * vd;
                                    }
                                }
                                dst[j] = gcta_grm_io::f32_to_fp16(diag_v);
                            } else {
                                for (int c = 0; c < j; ++c) {
                                    const float v = read_elem(is_identity ? c : kp[static_cast<size_t>(c)]);
                                    dst[c] = gcta_grm_io::f32_to_fp16(v);
                                    const double vd = static_cast<double>(v);
                                    row_off_sq += vd * vd;
                                }
                                dst[j] = gcta_grm_io::f32_to_fp16(diag_v);
                            }
                            out_diag[static_cast<size_t>(j)] = diag_v;
                            const double dd = static_cast<double>(diag_v);
                            tr_K  += dd;
                            tr_K2 += dd * dd + 2.0 * row_off_sq;
                        } else {
                            auto* dst = reinterpret_cast<float*>(wslot.data()) + dst_off;
                            if (is_identity && !in_is_f16) {
                                const auto* src = reinterpret_cast<const float*>(raw_ptr) + src_off;
                                std::memcpy(dst, src, static_cast<size_t>(j + 1) * sizeof(float));
                            } else if (!kp_runs.empty() && !in_is_f16) {
                                const auto* src = reinterpret_cast<const float*>(raw_ptr) + src_off;
                                for (const auto& run : kp_runs) {
                                    if (run.dst_start >= j) break;
                                    const int cnt = std::min(run.len, j - run.dst_start);
                                    std::memcpy(dst + run.dst_start, src + run.src_start,
                                                static_cast<size_t>(cnt) * sizeof(float));
                                }
                                dst[j] = diag_v;
                            } else {
                                for (int c = 0; c < j; ++c) {
                                    dst[c] = read_elem(is_identity ? c : kp[static_cast<size_t>(c)]);
                                }
                                dst[j] = diag_v;
                            }
                        }
                    }
                    reader.release_slot();
                    writer.submit_slot(out_bytes);
                }
                reader.stop();
            } else if (size_grm > 0) {
                // Non-monotonic (permuted) IDs: bounded-RAM 2D banded sequential streaming.
                // Zero random 4-byte fseeko+fread seeks.
                constexpr size_t kBandMemBudget = 512ULL << 20; // 512 MiB band buffer
                const int rows_per_band = std::clamp<int>(
                    static_cast<int>(kBandMemBudget / std::max<size_t>(1, static_cast<size_t>(size_grm) * sizeof(float))),
                    64, size_grm);

                std::vector<int> in_to_out_col(static_cast<size_t>(largest_grm_size), -1);
                for (int j = 0; j < size_grm; ++j) {
                    in_to_out_col[static_cast<size_t>(kp[static_cast<size_t>(j)])] = j;
                }
                std::vector<int> in_to_band_row(static_cast<size_t>(largest_grm_size), -1);
                std::vector<int> active_in_cols;
                active_in_cols.reserve(static_cast<size_t>(size_grm));
                for (int r = 0; r < largest_grm_size; ++r) {
                    if (in_to_out_col[static_cast<size_t>(r)] >= 0) active_in_cols.push_back(r);
                }

                const size_t max_in_elems = std::max<size_t>(1, gcta_grm_detail::kGrmAsyncSlotBytes / in_bpe);
                std::vector<gcta_grm_io::detail::RowChunkPlan> full_plans;
                std::vector<std::pair<off_t, size_t>> full_chunks;
                gcta_grm_io::detail::build_row_chunk_plan(
                    largest_grm_size, nullptr, /*is_identity=*/true, max_in_elems,
                    in_body_off, in_bpe, full_plans, full_chunks);

                AlignedRawBuffer<float, 64> band_buf(static_cast<size_t>(rows_per_band) * static_cast<size_t>(size_grm));

                for (int out_r0 = 0; out_r0 < size_grm; out_r0 += rows_per_band) {
                    const int out_r1 = std::min(size_grm, out_r0 + rows_per_band);
                    const int bh = out_r1 - out_r0;

                    std::fill(in_to_band_row.begin(), in_to_band_row.end(), -1);
                    std::vector<std::pair<int, int>> band_in_sorted;
                    band_in_sorted.reserve(static_cast<size_t>(bh));
                    for (int b = 0; b < bh; ++b) {
                        const int r_in = kp[static_cast<size_t>(out_r0 + b)];
                        in_to_band_row[static_cast<size_t>(r_in)] = b;
                        band_in_sorted.emplace_back(r_in, b);
                    }
                    std::sort(band_in_sorted.begin(), band_in_sorted.end());

                    gcta_io::AsyncSeqReader reader(in_fd, resolved.data_path, full_chunks,
                                                   /*max_slot_bytes=*/0, /*take_ownership=*/false);
                    const char* raw_ptr = nullptr;
                    size_t raw_len = 0;
                    size_t pidx = 0;
                    while (reader.next_slot(raw_ptr, raw_len)) {
                        const auto& plan = full_plans[pidx++];
                        const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;

                        #pragma omp parallel for schedule(dynamic, 64)
                        for (int r_in = plan.row_lo; r_in < plan.row_hi; ++r_in) {
                            const int b_r = in_to_band_row[static_cast<size_t>(r_in)];
                            const int o_c = in_to_out_col[static_cast<size_t>(r_in)];
                            if (b_r < 0 && o_c < 0) continue;

                            const size_t row_off = static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base;
                            auto get_v = [&](int c_in) -> float {
                                if (in_is_f16) {
                                    if (c_in == r_in) return in_diag[r_in];
                                    const auto* p = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + row_off;
                                    return gcta_grm_io::fp16_to_f32(p[c_in]);
                                }
                                const auto* p = reinterpret_cast<const float*>(raw_ptr) + row_off;
                                return p[c_in];
                            };

                            if (b_r >= 0) {
                                const int out_r = out_r0 + b_r;
                                float* dst_row = band_buf.data() + static_cast<size_t>(b_r) * static_cast<size_t>(size_grm);
                                for (int c_in : active_in_cols) {
                                    if (c_in > r_in) break;
                                    const int o_j = in_to_out_col[static_cast<size_t>(c_in)];
                                    if (o_j <= out_r) {
                                        dst_row[o_j] = get_v(c_in);
                                    }
                                }
                            }
                            if (o_c >= 0) {
                                for (const auto& [b_in, b] : band_in_sorted) {
                                    if (b_in >= r_in) break;
                                    const int out_r = out_r0 + b;
                                    if (o_c <= out_r) {
                                        band_buf[static_cast<size_t>(b) * static_cast<size_t>(size_grm) +
                                                 static_cast<size_t>(o_c)] = get_v(b_in);
                                    }
                                }
                            }
                        }
                        reader.release_slot();
                    }
                    reader.stop();

                    // Flush band [out_r0, out_r1) sequentially through AsyncSeqWriter.
                    const size_t max_out_elems = std::max<size_t>(1, gcta_grm_detail::kGrmAsyncSlotBytes / out_bpe);
                    int cur = out_r0;
                    while (cur < out_r1) {
                        const size_t b0 = static_cast<size_t>(cur) * (cur + 1) / 2;
                        const int next = gcta_grm_io::row_bound_for_cumulative(b0 + max_out_elems, cur, out_r1);
                        const size_t b1 = static_cast<size_t>(next) * (next + 1) / 2;
                        const size_t out_bytes = (b1 - b0) * out_bpe;

                        AlignedRawBuffer<char, 64>& wslot = writer.acquire_slot();
                        if (wslot.size() < out_bytes) wslot.resize(out_bytes);

                        #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
                        for (int r = cur; r < next; ++r) {
                            const float* src = band_buf.data() + static_cast<size_t>(r - out_r0) * static_cast<size_t>(size_grm);
                            const size_t dst_off = static_cast<size_t>(r) * (r + 1) / 2 - b0;
                            if (emit_f16) {
                                auto* dst = reinterpret_cast<gcta_grm_io::GrmFp16*>(wslot.data()) + dst_off;
                                double off_sq = 0.0;
                                #pragma omp simd reduction(+:off_sq)
                                for (int c = 0; c < r; ++c) {
                                    const float v = src[c];
                                    dst[c] = gcta_grm_io::f32_to_fp16(v);
                                    const double vd = static_cast<double>(v);
                                    off_sq += vd * vd;
                                }
                                const float dv = src[r];
                                dst[r] = gcta_grm_io::f32_to_fp16(dv);
                                out_diag[static_cast<size_t>(r)] = dv;
                                const double dd = static_cast<double>(dv);
                                tr_K  += dd;
                                tr_K2 += dd * dd + 2.0 * off_sq;
                            } else {
                                auto* dst = reinterpret_cast<float*>(wslot.data()) + dst_off;
                                std::memcpy(dst, src, static_cast<size_t>(r + 1) * sizeof(float));
                            }
                        }
                        writer.submit_slot(out_bytes);
                        cur = next;
                    }
                }
            }

            if (emit_f16) {
                gcta_grm_io::GrmFp16Header final_hdr{};
                std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
                final_hdr.version = gcta_grm_io::kGrmF16Version;
                final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
                final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
                final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized | gcta_grm_io::kGrmF16FlagHasTraces |
                                  gcta_grm_io::kGrmF16FlagHasDiagF32 |
                                  (in_is_f16 ? (in_hdr.flags & (gcta_grm_io::kGrmF16FlagZeroMissing |
                                                                gcta_grm_io::kGrmF16FlagVanRaden |
                                                                gcta_grm_io::kGrmF16FlagDominance |
                                                                gcta_grm_io::kGrmF16FlagHomogametic)) : 0u);
                final_hdr.n = static_cast<uint64_t>(size_grm);
                final_hdr.valid_markers = in_is_f16 ? in_hdr.valid_markers : 0ULL;
                final_hdr.m_snps_mean = in_is_f16 ? in_hdr.m_snps_mean : 0.0;
                final_hdr.id_hash = common_id_hash;
                final_hdr.trace_K = tr_K;
                final_hdr.trace_K2 = tr_K2;
                final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
                final_hdr.body_offset = out_body_off;
                final_hdr.part_row_start = 0;
                final_hdr.part_row_end = static_cast<uint64_t>(size_grm);
                final_hdr.mtd_weight = in_is_f16 ? in_hdr.mtd_weight : 1.0;
                gcta_grm_io::finalize_grm_f16_file(writer, final_hdr, out_diag.data(), static_cast<uint64_t>(size_grm));
            } else {
                writer.finish();
            }
        } catch (const std::exception& e) {
            ::close(in_fd);
            LOGGER.e(0, std::string("unify_grm failed on [") + resolved.data_path + "]: " + e.what());
        }

        ::close(in_fd);
        LOGGER.i(0, "GRM has been written to [" + wfile_name + "].");
    }
}

// ---------------------------------------------------------------------------
// GRM::prune_fam — grm sparse (sparse .grm.sp or --dense binary GRM)
// HPC-safe single-stream AsyncSeqReader + AsyncSeqWriter, zero std::endl,
// native .grm.f16 support.
// ---------------------------------------------------------------------------
void GRM::prune_fam(float thresh, bool isSparse, float *value){
    LOGGER.i(0, "Pruning the GRM to a " + std::string(isSparse ? "sparse" : "thresholded dense") +
                " matrix with a cutoff of " + to_string(thresh) + "...");

    const int n_in = static_cast<int>(num_subjects);
    const int n_out = static_cast<int>(index_keep.size());
    std::vector<int> kp(static_cast<size_t>(n_out));
    std::vector<string> keep_ID;
    keep_ID.reserve(static_cast<size_t>(n_out));
    bool is_identity = (n_in == n_out);
    for (int i = 0; i < n_out; ++i) {
        kp[static_cast<size_t>(i)] = static_cast<int>(index_keep[static_cast<size_t>(i)]);
        if (kp[static_cast<size_t>(i)] != i) is_identity = false;
        keep_ID.emplace_back(grm_ids[index_keep[static_cast<size_t>(i)]]);
    }

    LOGGER.i(2, "Saving " + to_string(keep_ID.size()) + " individual IDs");
    gcta_grm_io::write_id_file_buffered(o_name + ".grm.id", keep_ID);
    const uint64_t out_id_hash = gcta_io::hash_sample_ids(keep_ID);
    keep_ID.clear();
    keep_ID.shrink_to_fit();

    const bool in_is_f16 = (input_grm_format_ == gcta_grm_io::GrmDiskFormat::F16);
    const size_t in_bpe = in_is_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    const off_t in_body_off = in_is_f16 ? static_cast<off_t>(input_f16_hdr_.body_offset) : 0;

    std::vector<gcta_grm_io::KpRun> kp_runs;
    if (!is_identity && n_out > 0) {
        auto runs = gcta_grm_io::build_kp_runs(kp);
        if (runs.size() * 2 < static_cast<size_t>(n_out)) {
            kp_runs = std::move(runs);
        }
    }

    const size_t max_in_elems = std::max<size_t>(1, gcta_grm_detail::kGrmAsyncSlotBytes / in_bpe);
    std::vector<gcta_grm_io::detail::RowChunkPlan> plans;
    std::vector<std::pair<off_t, size_t>> io_chunks;
    gcta_grm_io::detail::build_row_chunk_plan(n_in, &kp, is_identity, max_in_elems,
                                              in_body_off, in_bpe, plans, io_chunks);

    const int in_fd = gcta_grm_io::open_readonly_checked(input_grm_data_path_);

    if (isSparse) {
        const std::string sp_path = o_name + ".grm.sp";
        auto sp_fd = gcta_grm_detail::open_posix_write_trunc(sp_path);
        if (!sp_fd) {
            ::close(in_fd);
            LOGGER.e(0, "can't write to [" + sp_path + "]");
        }
        const bool has_replace = (value != nullptr);
        const float rep_val = has_replace ? *value : 0.0f;
        uint64_t total_pairs = 0;

        try {
            gcta_io::AsyncSeqReader reader(in_fd, input_grm_data_path_, std::move(io_chunks),
                                           /*max_slot_bytes=*/0, /*take_ownership=*/true);
            const int n_thr = std::max(1, omp_get_max_threads());
            std::vector<std::string> thr_bufs(static_cast<size_t>(n_thr));
            std::string flush_buf;
            flush_buf.reserve(4ULL << 20);
            constexpr size_t kFlushBytes = 4ULL << 20;

            const char* raw_ptr = nullptr;
            size_t raw_len = 0;
            size_t pidx = 0;
            while (reader.next_slot(raw_ptr, raw_len)) {
                const auto& plan = plans[pidx++];
                const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
                const int j_lo = plan.j_lo;
                const int j_hi = plan.j_hi;
                const int rows = j_hi - j_lo;

                const int active_nt = (rows >= 64) ? std::min(n_thr, rows) : 1;
                uint64_t chunk_pairs = 0;
                #pragma omp parallel num_threads(active_nt) reduction(+:chunk_pairs)
                {
                    const int tid = omp_get_thread_num();
                    const int nt  = omp_get_num_threads();
                    const int r0 = j_lo + (rows * tid) / nt;
                    const int r1 = j_lo + (rows * (tid + 1)) / nt;
                    std::string& local = thr_bufs[static_cast<size_t>(tid)];
                    local.clear();

                    for (int j = r0; j < r1; ++j) {
                        const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                        const size_t row_off = static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base;
                        auto get_v = [&](int c_in) -> float {
                            if (in_is_f16) {
                                if (c_in == r_in) return input_f16_diag_[static_cast<size_t>(r_in)];
                                const auto* p = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + row_off;
                                return gcta_grm_io::fp16_to_f32(p[c_in]);
                            }
                            const auto* p = reinterpret_cast<const float*>(raw_ptr) + row_off;
                            return p[c_in];
                        };

                        for (int c = 0; c <= j; ++c) {
                            const int c_in = is_identity ? c : kp[static_cast<size_t>(c)];
                            const float v = get_v(c_in);
                            if (v > thresh) {
                                const float out_v = has_replace ? rep_val : v;
                                std::format_to(std::back_inserter(local), "{}\t{}\t{:.{}g}\n",
                                               j, c, out_v, std::numeric_limits<float>::digits10 + 2);
                                ++chunk_pairs;
                            }
                        }
                    }
                }
                total_pairs += chunk_pairs;
                reader.release_slot();

                for (int t = 0; t < active_nt; ++t) {
                    const std::string& tb = thr_bufs[static_cast<size_t>(t)];
                    if (tb.empty()) continue;
                    if (flush_buf.size() + tb.size() >= kFlushBytes) {
                        if (!flush_buf.empty()) {
                            gcta_io::write_exact_throwing(sp_fd.get(), flush_buf.data(), flush_buf.size(), sp_path);
                            flush_buf.clear();
                        }
                        if (tb.size() >= kFlushBytes) {
                            gcta_io::write_exact_throwing(sp_fd.get(), tb.data(), tb.size(), sp_path);
                        } else {
                            flush_buf.append(tb);
                        }
                    } else {
                        flush_buf.append(tb);
                    }
                }
            }
            reader.stop();
            if (!flush_buf.empty()) {
                gcta_io::write_exact_throwing(sp_fd.get(), flush_buf.data(), flush_buf.size(), sp_path);
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("prune_fam failed: ") + e.what());
        }

        sp_fd.reset();
        LOGGER.i(0, "Saving the sparse GRM (" + to_string(total_pairs) + " pairs) to [" + sp_path + "]");
        LOGGER.i(0, "Success:", "finished generating a sparse GRM");
        return;
    }

    // Dense thresholded output (--dense)
    const bool emit_f16 = write_fp16_ || in_is_f16;
    const size_t out_bpe = emit_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    const std::string out_path = o_name + (emit_f16 ? ".grm.f16" : ".grm.bin");
    const uint64_t out_body_off = emit_f16 ? gcta_grm_io::grm_f16_body_offset(static_cast<uint64_t>(n_out)) : 0ULL;
    const bool has_replace = (value != nullptr);
    const float rep_val = has_replace ? *value : 0.0f;

    std::vector<float> out_diag;
    if (emit_f16) out_diag.assign(static_cast<size_t>(n_out), 0.0f);
    double tr_K = 0.0, tr_K2 = 0.0;

    try {
        gcta_io::AsyncSeqReader reader(in_fd, input_grm_data_path_, std::move(io_chunks),
                                       /*max_slot_bytes=*/0, /*take_ownership=*/true);
        gcta_io::AsyncSeqWriter writer(out_path, gcta_grm_detail::kGrmAsyncSlotBytes,
                                       static_cast<off_t>(out_body_off));
        if (emit_f16) {
            gcta_grm_io::write_grm_f16_draft_prefix(writer, static_cast<uint64_t>(n_out));
        }

        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t pidx = 0;
        while (reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = plans[pidx++];
            const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
            const int j_lo = plan.j_lo;
            const int j_hi = plan.j_hi;
            const size_t out_base = static_cast<size_t>(j_lo) * (j_lo + 1) / 2;
            const size_t out_end  = static_cast<size_t>(j_hi) * (j_hi + 1) / 2;
            const size_t out_bytes = (out_end - out_base) * out_bpe;

            AlignedRawBuffer<char, 64>& wslot = writer.acquire_slot();
            if (wslot.size() < out_bytes) wslot.resize(out_bytes);

            #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
            for (int j = j_lo; j < j_hi; ++j) {
                const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                const size_t src_off = static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base;
                const size_t dst_off = static_cast<size_t>(j) * (j + 1) / 2 - out_base;
                auto get_v = [&](int c_in) -> float {
                    if (in_is_f16) {
                        if (c_in == r_in) return input_f16_diag_[static_cast<size_t>(r_in)];
                        const auto* p = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + src_off;
                        return gcta_grm_io::fp16_to_f32(p[c_in]);
                    }
                    const auto* p = reinterpret_cast<const float*>(raw_ptr) + src_off;
                    return p[c_in];
                };

                if (emit_f16) {
                    auto* dst = reinterpret_cast<gcta_grm_io::GrmFp16*>(wslot.data()) + dst_off;
                    double off_sq = 0.0;
                    for (int c = 0; c < j; ++c) {
                        const float raw_v = get_v(is_identity ? c : kp[static_cast<size_t>(c)]);
                        const float v = (raw_v > thresh) ? (has_replace ? rep_val : raw_v) : 0.0f;
                        dst[c] = gcta_grm_io::f32_to_fp16(v);
                        const double vd = static_cast<double>(v);
                        off_sq += vd * vd;
                    }
                    const float raw_d = get_v(r_in);
                    const float dv = (raw_d > thresh) ? (has_replace ? rep_val : raw_d) : 0.0f;
                    dst[j] = gcta_grm_io::f32_to_fp16(dv);
                    out_diag[static_cast<size_t>(j)] = dv;
                    const double dd = static_cast<double>(dv);
                    tr_K  += dd;
                    tr_K2 += dd * dd + 2.0 * off_sq;
                } else {
                    auto* dst = reinterpret_cast<float*>(wslot.data()) + dst_off;
                    for (int c = 0; c <= j; ++c) {
                        const float raw_v = get_v(is_identity ? c : kp[static_cast<size_t>(c)]);
                        dst[c] = (raw_v > thresh) ? (has_replace ? rep_val : raw_v) : 0.0f;
                    }
                }
            }
            reader.release_slot();
            writer.submit_slot(out_bytes);
        }
        reader.stop();

        if (emit_f16) {
            const bool zero_miss = in_is_f16 && ((input_f16_hdr_.flags & gcta_grm_io::kGrmF16FlagZeroMissing) != 0);
            gcta_grm_io::GrmFp16Header final_hdr{};
            std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
            final_hdr.version = gcta_grm_io::kGrmF16Version;
            final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
            final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized | gcta_grm_io::kGrmF16FlagHasTraces |
                              gcta_grm_io::kGrmF16FlagHasDiagF32 |
                              (zero_miss ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u);
            final_hdr.n = static_cast<uint64_t>(n_out);
            final_hdr.valid_markers = in_is_f16 ? input_f16_hdr_.valid_markers : 0ULL;
            final_hdr.m_snps_mean = in_is_f16 ? input_f16_hdr_.m_snps_mean : 0.0;
            final_hdr.id_hash = out_id_hash;
            final_hdr.trace_K = tr_K;
            final_hdr.trace_K2 = tr_K2;
            final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.body_offset = out_body_off;
            final_hdr.part_row_start = 0;
            final_hdr.part_row_end = static_cast<uint64_t>(n_out);
            final_hdr.mtd_weight = in_is_f16 ? input_f16_hdr_.mtd_weight : 1.0;
            gcta_grm_io::finalize_grm_f16_file(writer, final_hdr, out_diag.data(), static_cast<uint64_t>(n_out));
        } else {
            writer.finish();
        }
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("prune_fam dense write failed: ") + e.what());
    }
    LOGGER.i(0, "GRM has been saved to [" + out_path + "]");

    // Subset .grm.N.bin sequentially ONLY if needed (skip when emitting zero-missing .grm.f16).
    const bool in_zero_miss = in_is_f16 && ((input_f16_hdr_.flags & gcta_grm_io::kGrmF16FlagZeroMissing) != 0);
    if (emit_f16 && in_zero_miss) {
        return;
    }
    if (!input_has_n_file_) {
        if (!emit_f16) {
            LOGGER.error(0, "can't open [{}.grm.N.bin]", grm_file);
        }
        return;
    }

    const std::string in_n_path  = grm_file + ".grm.N.bin";
    const std::string out_n_path = o_name + ".grm.N.bin";
    const int n_in_fd = gcta_grm_io::open_readonly_checked(in_n_path);
    gcta_grm_io::detail::validate_f32_bin_size(n_in_fd, in_n_path, n_in);

    std::vector<gcta_grm_io::detail::RowChunkPlan> n_plans;
    std::vector<std::pair<off_t, size_t>> n_chunks;
    gcta_grm_io::detail::build_row_chunk_plan(n_in, &kp, is_identity,
                                              gcta_grm_detail::kGrmAsyncSlotBytes / sizeof(float),
                                              0, sizeof(float), n_plans, n_chunks);
    try {
        gcta_io::AsyncSeqReader n_reader(n_in_fd, in_n_path, std::move(n_chunks),
                                         /*max_slot_bytes=*/0, /*take_ownership=*/true);
        gcta_io::AsyncSeqWriter n_writer(out_n_path, gcta_grm_detail::kGrmAsyncSlotBytes);
        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t pidx = 0;
        while (n_reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = n_plans[pidx++];
            const auto* src_f32 = reinterpret_cast<const float*>(raw_ptr);
            const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
            const size_t out_base = static_cast<size_t>(plan.j_lo) * (plan.j_lo + 1) / 2;
            const size_t out_end  = static_cast<size_t>(plan.j_hi) * (plan.j_hi + 1) / 2;
            const size_t out_bytes = (out_end - out_base) * sizeof(float);

            AlignedRawBuffer<char, 64>& wslot = n_writer.acquire_slot();
            if (wslot.size() < out_bytes) wslot.resize(out_bytes);
            auto* dst_f32 = reinterpret_cast<float*>(wslot.data());

            #pragma omp parallel for schedule(dynamic, 64)
            for (int j = plan.j_lo; j < plan.j_hi; ++j) {
                const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                const float* srow = src_f32 + (static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base);
                float* drow = dst_f32 + (static_cast<size_t>(j) * (j + 1) / 2 - out_base);
                if (is_identity) {
                    std::memcpy(drow, srow, static_cast<size_t>(j + 1) * sizeof(float));
                } else if (!kp_runs.empty()) {
                    for (const auto& run : kp_runs) {
                        if (run.dst_start > j) break;
                        const int cnt = std::min(run.len, j + 1 - run.dst_start);
                        std::memcpy(drow + run.dst_start, srow + run.src_start,
                                    static_cast<size_t>(cnt) * sizeof(float));
                    }
                } else {
                    for (int c = 0; c <= j; ++c) drow[c] = srow[kp[static_cast<size_t>(c)]];
                }
            }
            n_reader.release_slot();
            n_writer.submit_slot(out_bytes);
        }
        n_reader.stop();
        n_writer.finish();
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("Failed writing pruned GRM N: ") + e.what());
    }
    LOGGER.i(0, "GRM N has been saved to [" + out_n_path + "]");
}

// ---------------------------------------------------------------------------
// GRM::cut_rel & write_pruned_grm_output — grm prune
// Fixes: O(U*E) std::count hang -> O(E) degree array, std::endl storm -> 4 MiB
// buffered writes, single-stream AsyncSeqReader/Writer, native .grm.f16 support.
// ---------------------------------------------------------------------------
void GRM::cut_rel(float thresh, bool no_grm){
    LOGGER.i(0, "Pruning the GRM with a cutoff of " + to_string(thresh) + "...");

    const int n_in = static_cast<int>(num_subjects);
    const int n_keep = static_cast<int>(index_keep.size());
    std::vector<int> kp(static_cast<size_t>(n_keep));
    bool is_identity = (n_in == n_keep);
    for (int i = 0; i < n_keep; ++i) {
        kp[static_cast<size_t>(i)] = static_cast<int>(index_keep[static_cast<size_t>(i)]);
        if (kp[static_cast<size_t>(i)] != i) is_identity = false;
    }

    const bool in_is_f16 = (input_grm_format_ == gcta_grm_io::GrmDiskFormat::F16);
    const size_t in_bpe = in_is_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    const off_t in_body_off = in_is_f16 ? static_cast<off_t>(input_f16_hdr_.body_offset) : 0;

    const size_t max_in_elems = std::max<size_t>(1, gcta_grm_detail::kGrmAsyncSlotBytes / in_bpe);
    std::vector<gcta_grm_io::detail::RowChunkPlan> plans;
    std::vector<std::pair<off_t, size_t>> io_chunks;
    gcta_grm_io::detail::build_row_chunk_plan(n_in, &kp, is_identity, max_in_elems,
                                              in_body_off, in_bpe, plans, io_chunks);

    vector<int> rm_grm_ID1, rm_grm_ID2;
    vector<float> rm_grm;

    const int in_fd = gcta_grm_io::open_readonly_checked(input_grm_data_path_);
    try {
        gcta_io::AsyncSeqReader reader(in_fd, input_grm_data_path_, std::move(io_chunks),
                                       /*max_slot_bytes=*/0, /*take_ownership=*/true);
        const int n_thr = std::max(1, omp_get_max_threads());
        std::vector<std::vector<int>> t_id1(static_cast<size_t>(n_thr));
        std::vector<std::vector<int>> t_id2(static_cast<size_t>(n_thr));
        std::vector<std::vector<float>> t_val(static_cast<size_t>(n_thr));

        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t pidx = 0;
        while (reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = plans[pidx++];
            const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
            const int j_lo = plan.j_lo;
            const int j_hi = plan.j_hi;
            const int rows = j_hi - j_lo;
            const int active_nt = (rows >= 64) ? std::min(n_thr, rows) : 1;

            #pragma omp parallel num_threads(active_nt)
            {
                const int tid = omp_get_thread_num();
                const int nt  = omp_get_num_threads();
                const int r0 = j_lo + (rows * tid) / nt;
                const int r1 = j_lo + (rows * (tid + 1)) / nt;
                auto& l1 = t_id1[static_cast<size_t>(tid)];
                auto& l2 = t_id2[static_cast<size_t>(tid)];
                auto& lv = t_val[static_cast<size_t>(tid)];
                l1.clear(); l2.clear(); lv.clear();

                for (int j = r0; j < r1; ++j) {
                    const int id1 = is_identity ? j : kp[static_cast<size_t>(j)];
                    const size_t row_off = static_cast<size_t>(id1) * (id1 + 1) / 2 - in_base;
                    if (in_is_f16) {
                        const auto* row = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + row_off;
                        for (int c = 0; c < j; ++c) {
                            const int id2 = is_identity ? c : kp[static_cast<size_t>(c)];
                            const float v = gcta_grm_io::fp16_to_f32(row[id2]);
                            if (v > thresh) {
                                l1.push_back(id1);
                                l2.push_back(id2);
                                lv.push_back(v);
                            }
                        }
                    } else {
                        const auto* row = reinterpret_cast<const float*>(raw_ptr) + row_off;
                        for (int c = 0; c < j; ++c) {
                            const int id2 = is_identity ? c : kp[static_cast<size_t>(c)];
                            const float v = row[id2];
                            if (v > thresh) {
                                l1.push_back(id1);
                                l2.push_back(id2);
                                lv.push_back(v);
                            }
                        }
                    }
                }
            }
            reader.release_slot();

            for (int t = 0; t < active_nt; ++t) {
                rm_grm_ID1.insert(rm_grm_ID1.end(), t_id1[static_cast<size_t>(t)].begin(), t_id1[static_cast<size_t>(t)].end());
                rm_grm_ID2.insert(rm_grm_ID2.end(), t_id2[static_cast<size_t>(t)].begin(), t_id2[static_cast<size_t>(t)].end());
                rm_grm.insert(rm_grm.end(), t_val[static_cast<size_t>(t)].begin(), t_val[static_cast<size_t>(t)].end());
            }
        }
        reader.stop();
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("cut_rel scan failed: ") + e.what());
    }

    if (cutoff_detail) {
        const std::string fam_path = o_name + ".family.txt";
        auto fam_fd = gcta_grm_detail::open_posix_write_trunc(fam_path);
        if (!fam_fd) {
            LOGGER.e(0, "can't write to [" + fam_path + "]");
        }
        constexpr size_t kFlushBytes = 4ULL << 20;
        std::string buf;
        buf.reserve(std::min<size_t>(kFlushBytes + 4096, rm_grm.size() * 48 + 4096));
        try {
            for (size_t idx = 0; idx < rm_grm.size(); ++idx) {
                std::format_to(std::back_inserter(buf), "{}\t{}\t{:g}\n",
                               grm_ids[static_cast<size_t>(rm_grm_ID1[idx])],
                               grm_ids[static_cast<size_t>(rm_grm_ID2[idx])],
                               rm_grm[idx]);
                if (buf.size() >= kFlushBytes) {
                    gcta_io::write_exact_throwing(fam_fd.get(), buf.data(), buf.size(), fam_path);
                    buf.clear();
                }
            }
            if (!buf.empty()) {
                gcta_io::write_exact_throwing(fam_fd.get(), buf.data(), buf.size(), fam_path);
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("cut_rel failed writing [") + fam_path + "]: " + e.what());
        }
        fam_fd.reset();
        LOGGER.i(0, "Related family pairs have been saved to " + fam_path);
    }

    // O(E) degree histogram (replaces O(U*E) std::count scan from legacy GCTA 1.26; exact same tie behavior).
    std::vector<int> deg(static_cast<size_t>(num_subjects), 0);
    for (size_t i = 0; i < rm_grm_ID1.size(); ++i) {
        deg[static_cast<size_t>(rm_grm_ID1[i])]++;
        deg[static_cast<size_t>(rm_grm_ID2[i])]++;
    }
    for (size_t i = 0; i < rm_grm_ID1.size(); ++i) {
        if (deg[static_cast<size_t>(rm_grm_ID1[i])] < deg[static_cast<size_t>(rm_grm_ID2[i])]) {
            std::swap(rm_grm_ID1[i], rm_grm_ID2[i]);
        }
    }

    std::sort(rm_grm_ID1.begin(), rm_grm_ID1.end());
    rm_grm_ID1.erase(std::unique(rm_grm_ID1.begin(), rm_grm_ID1.end()), rm_grm_ID1.end());

    auto diff = [&rm_grm_ID1](uint32_t value) -> bool {
        return std::binary_search(rm_grm_ID1.begin(), rm_grm_ID1.end(), static_cast<int>(value));
    };
    std::erase_if(index_keep, diff);

    vector<string> keep_ID;
    keep_ID.reserve(index_keep.size());
    for (uint32_t idx : index_keep) {
        keep_ID.emplace_back(grm_ids[idx]);
    }

    LOGGER.i(0, "After pruning the GRM, there are " + to_string(keep_ID.size()) +
                " individuals (" + to_string(rm_grm_ID1.size()) + " individuals removed).");
    if (!no_grm) {
        gcta_grm_io::write_id_file_buffered(o_name + ".grm.id", keep_ID);
        LOGGER.i(2, "Pruned unrelated IDs have been saved to " + o_name + ".grm.id");
    }
    if (cutoff_detail) {
        gcta_grm_io::write_id_file_buffered(o_name + ".singleton.txt", keep_ID);
        LOGGER.i(2, "Pruned singleton IDs have been saved to " + o_name + ".singleton.txt");
    }

    if (no_grm) {
        return;
    }

    const bool emit_f16 = write_fp16_ || in_is_f16;
    write_pruned_grm_output(emit_f16);
}

void GRM::write_pruned_grm_output(bool out_fp16) {
    const int n_in = static_cast<int>(num_subjects);
    const int n_out = static_cast<int>(index_keep.size());
    std::vector<int> kp(static_cast<size_t>(n_out));
    std::vector<string> keep_ids;
    keep_ids.reserve(static_cast<size_t>(n_out));
    bool is_identity = (n_in == n_out);
    for (int i = 0; i < n_out; ++i) {
        kp[static_cast<size_t>(i)] = static_cast<int>(index_keep[static_cast<size_t>(i)]);
        if (kp[static_cast<size_t>(i)] != i) is_identity = false;
        keep_ids.push_back(grm_ids[index_keep[static_cast<size_t>(i)]]);
    }
    const uint64_t out_id_hash = gcta_io::hash_sample_ids(keep_ids);

    std::vector<gcta_grm_io::KpRun> kp_runs;
    if (!is_identity && n_out > 0) {
        auto runs = gcta_grm_io::build_kp_runs(kp);
        if (runs.size() * 2 < static_cast<size_t>(n_out)) {
            kp_runs = std::move(runs);
        }
    }

    const bool in_is_f16 = (input_grm_format_ == gcta_grm_io::GrmDiskFormat::F16);
    const size_t in_bpe = in_is_f16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    const size_t out_bpe = out_fp16 ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
    const off_t in_body_off = in_is_f16 ? static_cast<off_t>(input_f16_hdr_.body_offset) : 0;
    const std::string out_path = o_name + (out_fp16 ? ".grm.f16" : ".grm.bin");
    const uint64_t out_body_off = out_fp16 ? gcta_grm_io::grm_f16_body_offset(static_cast<uint64_t>(n_out)) : 0ULL;

    std::vector<gcta_grm_io::detail::RowChunkPlan> plans;
    std::vector<std::pair<off_t, size_t>> io_chunks;
    gcta_grm_io::detail::build_row_chunk_plan(n_in, &kp, is_identity,
                                              gcta_grm_detail::kGrmAsyncSlotBytes / in_bpe,
                                              in_body_off, in_bpe, plans, io_chunks);

    std::vector<float> out_diag;
    if (out_fp16) out_diag.assign(static_cast<size_t>(n_out), 0.0f);
    double tr_K = 0.0, tr_K2 = 0.0;

    LOGGER.i(0, "Writing pruned GRM values to [" + out_path + "]...");
    const int in_fd = gcta_grm_io::open_readonly_checked(input_grm_data_path_);
    try {
        gcta_io::AsyncSeqReader reader(in_fd, input_grm_data_path_, std::move(io_chunks),
                                       /*max_slot_bytes=*/0, /*take_ownership=*/true);
        gcta_io::AsyncSeqWriter writer(out_path, gcta_grm_detail::kGrmAsyncSlotBytes,
                                       static_cast<off_t>(out_body_off));
        if (out_fp16) {
            gcta_grm_io::write_grm_f16_draft_prefix(writer, static_cast<uint64_t>(n_out));
        }

        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t pidx = 0;
        while (reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = plans[pidx++];
            const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
            const size_t out_base = static_cast<size_t>(plan.j_lo) * (plan.j_lo + 1) / 2;
            const size_t out_end  = static_cast<size_t>(plan.j_hi) * (plan.j_hi + 1) / 2;
            const size_t out_bytes = (out_end - out_base) * out_bpe;

            AlignedRawBuffer<char, 64>& wslot = writer.acquire_slot();
            if (wslot.size() < out_bytes) wslot.resize(out_bytes);

            #pragma omp parallel for reduction(+:tr_K, tr_K2) schedule(dynamic, 64)
            for (int j = plan.j_lo; j < plan.j_hi; ++j) {
                const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                const size_t src_off = static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base;
                const size_t dst_off = static_cast<size_t>(j) * (j + 1) / 2 - out_base;

                if (out_fp16) {
                    auto* dst = reinterpret_cast<gcta_grm_io::GrmFp16*>(wslot.data()) + dst_off;
                    double off_sq = 0.0;
                    if (in_is_f16) {
                        const auto* srow = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + src_off;
                        if (is_identity) {
                            std::memcpy(dst, srow, static_cast<size_t>(j + 1) * sizeof(gcta_grm_io::GrmFp16));
                            #pragma omp simd reduction(+:off_sq)
                            for (int c = 0; c < j; ++c) {
                                const double vd = static_cast<double>(gcta_grm_io::fp16_to_f32(srow[c]));
                                off_sq += vd * vd;
                            }
                        } else if (!kp_runs.empty()) {
                            for (const auto& run : kp_runs) {
                                if (run.dst_start >= j) break;
                                const int cnt = std::min(run.len, j - run.dst_start);
                                std::memcpy(dst + run.dst_start, srow + run.src_start,
                                            static_cast<size_t>(cnt) * sizeof(gcta_grm_io::GrmFp16));
                                #pragma omp simd reduction(+:off_sq)
                                for (int k = 0; k < cnt; ++k) {
                                    const double vd = static_cast<double>(gcta_grm_io::fp16_to_f32(srow[run.src_start + k]));
                                    off_sq += vd * vd;
                                }
                            }
                            dst[j] = srow[r_in];
                        } else {
                            for (int c = 0; c < j; ++c) {
                                const auto h = srow[kp[static_cast<size_t>(c)]];
                                dst[c] = h;
                                const double vd = static_cast<double>(gcta_grm_io::fp16_to_f32(h));
                                off_sq += vd * vd;
                            }
                            dst[j] = srow[r_in];
                        }
                        const float dv = input_f16_diag_[static_cast<size_t>(r_in)];
                        out_diag[static_cast<size_t>(j)] = dv;
                        const double dd = static_cast<double>(dv);
                        tr_K  += dd;
                        tr_K2 += dd * dd + 2.0 * off_sq;
                    } else {
                        const auto* srow = reinterpret_cast<const float*>(raw_ptr) + src_off;
                        for (int c = 0; c < j; ++c) {
                            const float v = srow[is_identity ? c : kp[static_cast<size_t>(c)]];
                            dst[c] = gcta_grm_io::f32_to_fp16(v);
                            const double vd = static_cast<double>(v);
                            off_sq += vd * vd;
                        }
                        const float dv = srow[r_in];
                        dst[j] = gcta_grm_io::f32_to_fp16(dv);
                        out_diag[static_cast<size_t>(j)] = dv;
                        const double dd = static_cast<double>(dv);
                        tr_K  += dd;
                        tr_K2 += dd * dd + 2.0 * off_sq;
                    }
                } else {
                    auto* dst = reinterpret_cast<float*>(wslot.data()) + dst_off;
                    if (in_is_f16) {
                        const auto* srow = reinterpret_cast<const gcta_grm_io::GrmFp16*>(raw_ptr) + src_off;
                        for (int c = 0; c < j; ++c) {
                            dst[c] = gcta_grm_io::fp16_to_f32(srow[is_identity ? c : kp[static_cast<size_t>(c)]]);
                        }
                        dst[j] = input_f16_diag_[static_cast<size_t>(r_in)];
                    } else {
                        const auto* srow = reinterpret_cast<const float*>(raw_ptr) + src_off;
                        if (is_identity) {
                            std::memcpy(dst, srow, static_cast<size_t>(j + 1) * sizeof(float));
                        } else if (!kp_runs.empty()) {
                            for (const auto& run : kp_runs) {
                                if (run.dst_start > j) break;
                                const int cnt = std::min(run.len, j + 1 - run.dst_start);
                                std::memcpy(dst + run.dst_start, srow + run.src_start,
                                            static_cast<size_t>(cnt) * sizeof(float));
                            }
                        } else {
                            for (int c = 0; c <= j; ++c) dst[c] = srow[kp[static_cast<size_t>(c)]];
                        }
                    }
                }
            }
            reader.release_slot();
            writer.submit_slot(out_bytes);
        }
        reader.stop();

        if (out_fp16) {
            const bool zero_miss = in_is_f16 && ((input_f16_hdr_.flags & gcta_grm_io::kGrmF16FlagZeroMissing) != 0);
            gcta_grm_io::GrmFp16Header final_hdr{};
            std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
            final_hdr.version = gcta_grm_io::kGrmF16Version;
            final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
            final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized | gcta_grm_io::kGrmF16FlagHasTraces |
                              gcta_grm_io::kGrmF16FlagHasDiagF32 |
                              (zero_miss ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u);
            final_hdr.n = static_cast<uint64_t>(n_out);
            final_hdr.valid_markers = in_is_f16 ? input_f16_hdr_.valid_markers : 0ULL;
            final_hdr.m_snps_mean = in_is_f16 ? input_f16_hdr_.m_snps_mean : 0.0;
            final_hdr.id_hash = out_id_hash;
            final_hdr.trace_K = tr_K;
            final_hdr.trace_K2 = tr_K2;
            final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
            final_hdr.body_offset = out_body_off;
            final_hdr.part_row_start = 0;
            final_hdr.part_row_end = static_cast<uint64_t>(n_out);
            final_hdr.mtd_weight = in_is_f16 ? input_f16_hdr_.mtd_weight : 1.0;
            gcta_grm_io::finalize_grm_f16_file(writer, final_hdr, out_diag.data(), static_cast<uint64_t>(n_out));
        } else {
            writer.finish();
        }
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("write_pruned_grm_output failed: ") + e.what());
    }
    LOGGER.i(2, "GRM values have been saved to [" + out_path + "]");

    const bool in_zero_miss = in_is_f16 && ((input_f16_hdr_.flags & gcta_grm_io::kGrmF16FlagZeroMissing) != 0);
    if (out_fp16 && in_zero_miss) {
        return;
    }
    if (!input_has_n_file_) {
        if (!out_fp16) LOGGER.w(2, "There is no [" + grm_file + ".grm.N.bin]");
        return;
    }

    const std::string in_n_path  = grm_file + ".grm.N.bin";
    const std::string out_n_path = o_name + ".grm.N.bin";
    const int n_in_fd = gcta_grm_io::open_readonly_checked(in_n_path);
    gcta_grm_io::detail::validate_f32_bin_size(n_in_fd, in_n_path, n_in);

    std::vector<gcta_grm_io::detail::RowChunkPlan> n_plans;
    std::vector<std::pair<off_t, size_t>> n_chunks;
    gcta_grm_io::detail::build_row_chunk_plan(n_in, &kp, is_identity,
                                              gcta_grm_detail::kGrmAsyncSlotBytes / sizeof(float),
                                              0, sizeof(float), n_plans, n_chunks);
    try {
        gcta_io::AsyncSeqReader n_reader(n_in_fd, in_n_path, std::move(n_chunks),
                                         /*max_slot_bytes=*/0, /*take_ownership=*/true);
        gcta_io::AsyncSeqWriter n_writer(out_n_path, gcta_grm_detail::kGrmAsyncSlotBytes);
        const char* raw_ptr = nullptr;
        size_t raw_len = 0;
        size_t pidx = 0;
        while (n_reader.next_slot(raw_ptr, raw_len)) {
            const auto& plan = n_plans[pidx++];
            const auto* src_f32 = reinterpret_cast<const float*>(raw_ptr);
            const size_t in_base = static_cast<size_t>(plan.row_lo) * (plan.row_lo + 1) / 2;
            const size_t out_base = static_cast<size_t>(plan.j_lo) * (plan.j_lo + 1) / 2;
            const size_t out_end  = static_cast<size_t>(plan.j_hi) * (plan.j_hi + 1) / 2;
            const size_t out_bytes = (out_end - out_base) * sizeof(float);

            AlignedRawBuffer<char, 64>& wslot = n_writer.acquire_slot();
            if (wslot.size() < out_bytes) wslot.resize(out_bytes);
            auto* dst_f32 = reinterpret_cast<float*>(wslot.data());

            #pragma omp parallel for schedule(dynamic, 64)
            for (int j = plan.j_lo; j < plan.j_hi; ++j) {
                const int r_in = is_identity ? j : kp[static_cast<size_t>(j)];
                const float* srow = src_f32 + (static_cast<size_t>(r_in) * (r_in + 1) / 2 - in_base);
                float* drow = dst_f32 + (static_cast<size_t>(j) * (j + 1) / 2 - out_base);
                if (is_identity) {
                    std::memcpy(drow, srow, static_cast<size_t>(j + 1) * sizeof(float));
                } else if (!kp_runs.empty()) {
                    for (const auto& run : kp_runs) {
                        if (run.dst_start > j) break;
                        const int cnt = std::min(run.len, j + 1 - run.dst_start);
                        std::memcpy(drow + run.dst_start, srow + run.src_start,
                                    static_cast<size_t>(cnt) * sizeof(float));
                    }
                } else {
                    for (int c = 0; c <= j; ++c) drow[c] = srow[kp[static_cast<size_t>(c)]];
                }
            }
            n_reader.release_slot();
            n_writer.submit_slot(out_bytes);
        }
        n_reader.stop();
        n_writer.finish();
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("Failed writing pruned GRM N: ") + e.what());
    }
    LOGGER.i(2, "Number of SNPs has been saved to [" + out_n_path + "]");
}

// ---------------------------------------------------------------------------
// GRM builder-mode constructor (grm make)
// ---------------------------------------------------------------------------
GRM::GRM(Pheno* pheno, Marker* marker, const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix) {
    n_markers_explicit_ = opts.n_markers.has_value();
    write_fp16_         = opts.grm_fp16;
    fast_preset_        = opts.fast;
    is_homogametic_     = opts.homogametic;

    this->pheno = pheno;
    this->marker = marker;
    this->geno = std::make_unique<Geno>(pheno, marker, opts.geno, opts.marker_filter, out_prefix);
    this->index_keep = pheno->get_index_keep();
    this->num_parts = opts.part ? static_cast<uint32_t>(opts.part->first) : 1u;
    this->part = opts.part ? static_cast<uint32_t>(opts.part->second) : 1u;

    bBLAS = true;
    vector<uint32_t> parts = divide_parts_mem(pheno->count_keep(), num_parts);

    if (parts.size() != num_parts) {
        LOGGER.w(0, "cannot divide into " + to_string(num_parts) + ". Use " + to_string(parts.size()) + " instead.");
        this->num_parts = parts.size();
    }

    if (part > this->num_parts) {
        LOGGER.e(0, "cannot calculate when" + to_string(part) + " is larger than " + to_string(this->num_parts));
    } else {
        if (part == 1) {
            part_keep_indices = std::make_pair(0, parts[0]);
        } else {
            part_keep_indices = std::make_pair(parts[part - 2] + 1, parts[part - 1]);
        }
    }

    this->num_count_handle = 2;

    num_individual = part_keep_indices.second - part_keep_indices.first + 1;
    num_grm = ((uint64_t) part_keep_indices.first + part_keep_indices.second + 2) * num_individual / 2;

    grm_tile_budget_bytes = opts.tile_budget_gb
        ? (*opts.tile_budget_gb * 1024.0 * 1024.0 * 1024.0)
        : 0.0;
    if (opts.sparse_cutoff.has_value()) {
        sparse_cutoff = static_cast<float>(*opts.sparse_cutoff);
    }

    sub_miss.assign(index_keep.size() + 64, 0);

    int num_thread = omp_get_max_threads();
    index_grm_pairs.reserve(num_thread);
    vector<uint32_t> thread_parts = divide_parts(part_keep_indices.first, part_keep_indices.second, num_thread);
    if (num_thread != static_cast<int>(thread_parts.size())) {
        LOGGER.w(0, "cannot run in " + to_string(num_thread) + " threads. Use " + to_string(thread_parts.size()) + " instead");
    }

    index_grm_pairs.push_back(std::make_pair(static_cast<int>(part_keep_indices.first), static_cast<int>(thread_parts[0])));
    for (size_t index = 1; index != thread_parts.size(); index++) {
        index_grm_pairs.push_back(std::make_pair(static_cast<int>(thread_parts[index - 1] + 1), static_cast<int>(thread_parts[index])));
    }

    isDominance = opts.dominance;
    isMtd = (opts.alg == gcta_cli::GrmAlg::VanRaden);

    if (fast_preset_) {
        std::string msg = "Using --fast preset:";
        if (!opts.alg_explicit && isMtd) msg += " --alg vanraden (INT8 VNNI)";
        if (write_fp16_) msg += " --grm-fp16 (native .grm.f16, exact float32 diagonal, zero-missing .grm.N.bin omitted)";
        LOGGER.i(0, msg + ".");
    }

    grm_n = static_cast<int>(part_keep_indices.second) + 1;
    grm_m = static_cast<int>(part_keep_indices.second) - static_cast<int>(part_keep_indices.first) + 1;
    grm_s_n = grm_n - grm_m;
    grm_bytes_std_geno = sizeof(float) * static_cast<size_t>(grm_n);
    const int ld_base = std::max(grm_n, static_cast<int>(index_keep.size()));
    stdGenoLD = (ld_base + 63) / 64 * 64;

    const bool auto_use_int8 = bBLAS && isMtd && !isDominance && !opts.homogametic
                               && (geno->getGenoFormat() == "BED")
                               && (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);
    if (n_markers_explicit_) {
        GRM::nMarkerBlock = *opts.n_markers;
    } else if (auto_use_int8) {
        constexpr uint64_t kTargetKMajorBytes = 2ULL << 30;
        const uint64_t raw_k = kTargetKMajorBytes / std::max<uint64_t>(1ULL, static_cast<uint64_t>(grm_n));
        GRM::nMarkerBlock = static_cast<int>(std::clamp<uint64_t>((raw_k / 64ULL) * 64ULL, 1024ULL, 4096ULL));
    } else {
        GRM::nMarkerBlock = 1024;
    }

    validIndexBuf.reserve(static_cast<size_t>(nMarkerBlock));

    string fstring = bBLAS ? " v2 " : " ";
    string com_string = string("Computing the ") + (isDominance ? "dominance " : "") + "genetic relationship matrix (GRM)" + fstring + " using " + std::to_string(nMarkerBlock) + " markers per block ...";
    LOGGER.i(0, com_string);
    LOGGER.i(0, "Subset " + to_string(part) + "/" + to_string(num_parts) + ", no. subject " + to_string(part_keep_indices.first + 1) + "-" + to_string(part_keep_indices.second + 1));
    LOGGER.i(1, to_string(num_individual) + " samples, " + to_string(marker->count_extract()) + " markers, " + to_string(num_grm) + " GRM elements");

    o_name = out_prefix;
    if (isDominance) {
        o_name += ".d";
    }

    output_id();
}

void GRM::output_id() {
    vector<string> out_id = pheno->get_id(part_keep_indices.first, part_keep_indices.second);
    grm_id_hash_ = gcta_io::hash_sample_ids(out_id);
    const string o_grm_id = o_name + ".grm.id";
    gcta_grm_io::write_id_file_buffered(o_grm_id, out_id);
    LOGGER.i(0, "IDs for the GRM file have been saved in the file [" + o_grm_id + "]");
}

// ---------------------------------------------------------------------------
// 4096-row strip-packed GRM allocation, strip-by-strip promotion & unified packed N
// ---------------------------------------------------------------------------
void GRM::reset_grm_strips() {
    grm_strips_.clear();
    grm_i32_strips_.clear();
}

void GRM::allocate_grm_strips(bool as_i32) {
    reset_grm_strips();
    const int num_strips = (grm_tile_rows + kGrmStripRows - 1) / kGrmStripRows;
    if (as_i32) {
        grm_i32_strips_.resize(static_cast<size_t>(num_strips));
        for (int s = 0; s < num_strips; ++s) {
            const int r0 = s * kGrmStripRows;
            const int rows_s = std::min(kGrmStripRows, grm_tile_rows - r0);
            const int cols_s = grm_tile_rs + r0 + rows_s;
            const size_t elems = static_cast<size_t>(rows_s) * static_cast<size_t>(cols_s);
            grm_i32_strips_[static_cast<size_t>(s)] = make_aligned_uninitialized<int32_t, 64>(elems);
            if (!grm_i32_strips_[static_cast<size_t>(s)]) {
                reset_grm_strips();
                LOGGER.error(0, "can't allocate enough memory for strip-packed int32 GRM (strip {}/{}).", s + 1, num_strips);
            }
        }
    } else {
        grm_strips_.resize(static_cast<size_t>(num_strips));
        for (int s = 0; s < num_strips; ++s) {
            const int r0 = s * kGrmStripRows;
            const int rows_s = std::min(kGrmStripRows, grm_tile_rows - r0);
            const int cols_s = grm_tile_rs + r0 + rows_s;
            const size_t elems = static_cast<size_t>(rows_s) * static_cast<size_t>(cols_s);
            grm_strips_[static_cast<size_t>(s)] = make_aligned_uninitialized<double, 64>(elems);
            if (!grm_strips_[static_cast<size_t>(s)]) {
                reset_grm_strips();
                LOGGER.error(0, "can't allocate enough memory for strip-packed double GRM (strip {}/{}).", s + 1, num_strips);
            }
        }
    }
}

void GRM::ensure_N_allocated() {
    if (N) return;
    // Unified packed lower-triangular layout for [grm_tile_rs, grm_tile_re) in BOTH tiled and dense paths.
    const uint64_t tile_tri =
        (static_cast<uint64_t>(grm_tile_re) + static_cast<uint64_t>(grm_tile_rs) + 1ULL) *
        static_cast<uint64_t>(grm_tile_rows) / 2ULL;
    const uint64_t fill_N = (tile_tri + static_cast<uint64_t>(num_count_handle) - 1ULL) /
                            static_cast<uint64_t>(num_count_handle) * static_cast<uint64_t>(num_count_handle);
    N = make_aligned_uninitialized<uint32_t, 64>(fill_N);
    if (!N) {
        LOGGER.error(0, "can't allocate enough memory to store packed N: {:.2f}GB required.",
                     fill_N * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0));
    }
    uint32_t* const N_ptr = std::assume_aligned<64>(N.get());
    const uint64_t rs_u = static_cast<uint64_t>(grm_tile_rs);
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < static_cast<int>(index_grm_pairs.size()); ++idx) {
        const auto [p_from, p_to] = index_grm_pairs[static_cast<size_t>(idx)];
        const uint64_t off_start = (static_cast<uint64_t>(p_from) + 1ULL + rs_u)
                                 * static_cast<uint64_t>(p_from - grm_tile_rs) / 2ULL;
        const uint64_t off_end   = (static_cast<uint64_t>(p_to) + 2ULL + rs_u)
                                 * static_cast<uint64_t>(p_to + 1 - grm_tile_rs) / 2ULL;
        std::memset(N_ptr + off_start, 0, static_cast<size_t>(off_end - off_start) * sizeof(uint32_t));
    }
    if (fill_N > tile_tri) {
        std::memset(N_ptr + tile_tri, 0, static_cast<size_t>(fill_N - tile_tri) * sizeof(uint32_t));
    }
}

void GRM::promote_i32_to_f64() {
    if (!grm_is_i32_) return;
    const int num_strips = (grm_tile_rows + kGrmStripRows - 1) / kGrmStripRows;
    if (grm_first_accum) {
        // Nothing accumulated into grm_i32_strips_ yet: free first so peak RSS stays bounded.
        grm_i32_strips_.clear();
        allocate_grm_strips(/*as_i32=*/false);
    } else {
        // Strip-by-strip promotion: allocate double strip s, convert from int32 strip s,
        // and immediately release int32 strip s (eliminates transient 12 B/elem peak RSS spike!).
        grm_strips_.resize(static_cast<size_t>(num_strips));
        const double* const R_ptr = R_total_.data();
        const double S_tot = S_total_;

        for (int s = 0; s < num_strips; ++s) {
            const int r0 = s * kGrmStripRows;
            const int rows_s = std::min(kGrmStripRows, grm_tile_rows - r0);
            const int global_r0 = grm_tile_rs + r0;
            const int cols_s = global_r0 + rows_s;
            const size_t elems = static_cast<size_t>(rows_s) * static_cast<size_t>(cols_s);

            grm_strips_[static_cast<size_t>(s)] = make_aligned_uninitialized<double, 64>(elems);
            if (!grm_strips_[static_cast<size_t>(s)]) {
                LOGGER.error(0, "can't allocate enough memory to promote strip {} to double.", s);
            }

            const int32_t* const src_base = std::assume_aligned<64>(grm_i32_strips_[static_cast<size_t>(s)].get());
            double* const dst_base        = std::assume_aligned<64>(grm_strips_[static_cast<size_t>(s)].get());
            const double* const R_row     = R_ptr + global_r0;

            if (global_r0 > 0) {
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < global_r0; ++c) {
                    const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * rows_s;
                    double* __restrict__ dst_col        = dst_base + static_cast<size_t>(c) * rows_s;
                    const double r_j_plus_S = S_tot - R_ptr[c];
                    #pragma omp simd
                    for (int r = 0; r < rows_s; ++r) {
                        dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                    }
                }
            }
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(rows_s, tid, nth);
                for (int c_local = c_beg; c_local < c_end; ++c_local) {
                    const int c = global_r0 + c_local;
                    const int32_t* __restrict__ src_col = src_base + static_cast<size_t>(c) * rows_s;
                    double* __restrict__ dst_col        = dst_base + static_cast<size_t>(c) * rows_s;
                    const double r_j_plus_S = S_tot - R_ptr[c];
                    #pragma omp simd
                    for (int r = c_local; r < rows_s; ++r) {
                        dst_col[r] = static_cast<double>(src_col[r]) - R_row[r] + r_j_plus_S;
                    }
                }
            }
            grm_i32_strips_[static_cast<size_t>(s)].reset();
        }
        grm_i32_strips_.clear();
    }
    grm_is_i32_ = false;
    const int max_tile_dim = std::min(kGrmStripRows, std::max(1, grm_n));
    const size_t tile_elems = static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim);
    if (grm_i32_tile_.size() < tile_elems) {
        grm_i32_tile_.resize(tile_elems);
    }
}

void GRM::calculate_GRM_blas(uintptr_t *buf, std::span<const uint32_t> markerIndex){
    const int num_marker = static_cast<int>(markerIndex.size());
    const uint32_t full_keep_ct = static_cast<uint32_t>(index_keep.size());

    if (use_int8_grm && use_tile1_cache_) {
        // Tiles 2..T cached fast path: skip redundant stats_only VPOPCNTDQ pass completely!
        if (tile1_cursor_ + static_cast<size_t>(num_marker) > tile1_valid_extract_.size()) {
            LOGGER.e(0, "Internal error: Tile-1 marker cache cursor overflow.");
        }
        validIndexBuf.clear();
        for (int i = 0; i < num_marker; ++i) {
            const size_t ci = tile1_cursor_ + static_cast<size_t>(i);
            if (markerIndex[static_cast<size_t>(i)] != tile1_valid_extract_[ci]) {
                LOGGER.e(0, "Internal error: Tile-1 marker cache index mismatch.");
            }
            GenoBufItem& item = gbufitems[static_cast<size_t>(i)];
            item.extractedMarkerIndex = markerIndex[static_cast<size_t>(i)];
            item.geno_ptr = nullptr;
            item.geno_f32_ptr = nullptr;
            item.geno_i8_ptr = nullptr;
            item.valid = true;
            item.mean = tile1_valid_mean_[ci];
            item.nValidN = (tile1_valid_has_miss_[ci] != 0)
                ? (full_keep_ct > 0 ? full_keep_ct - 1u : 0u)
                : full_keep_ct;
            validIndexBuf.push_back(i);
        }
        tile1_cursor_ += static_cast<size_t>(num_marker);
    } else if (use_int8_grm) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_marker; i++) {
            GenoBufItem &item = gbufitems[static_cast<size_t>(i)];
            item.extractedMarkerIndex = markerIndex[static_cast<size_t>(i)];
            item.geno_ptr = nullptr;
            item.geno_f32_ptr = nullptr;
            item.geno_i8_ptr = nullptr;
            item.stats_only = true;
            item.lazy_missing = true;
            geno->getGenoDouble(buf, i, &item);
        }
        validIndexBuf.clear();
        for (int i = 0; i < num_marker; i++) {
            if (gbufitems[static_cast<size_t>(i)].valid) {
                validIndexBuf.push_back(i);
            }
        }
    } else {
        float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_marker; i++) {
            GenoBufItem &item = gbufitems[static_cast<size_t>(i)];
            item.extractedMarkerIndex = markerIndex[static_cast<size_t>(i)];
            item.geno_ptr = nullptr;
            item.geno_i8_ptr = nullptr;
            item.geno_f32_ptr = std_geno_base + static_cast<size_t>(i) * stdGenoLD;
            item.stats_only = false;
            item.lazy_missing = true;
            geno->getGenoDouble(buf, i, &item);
        }
        validIndexBuf.clear();
        for (int i = 0; i < num_marker; i++) {
            if (gbufitems[static_cast<size_t>(i)].valid) {
                validIndexBuf.push_back(i);
            }
        }
    }

    const int curNumValidMarkers = static_cast<int>(validIndexBuf.size());

    if (curNumValidMarkers == 0) {
        if (!grm_skip_global_state) {
            finished_marker += num_marker;
        }
        return;
    }

    // Record Tile-1 valid markers & stats for Tiles 2..T streaming reuse (INT8 BED path).
    if (use_int8_grm && grm_tiling_enabled && !grm_skip_global_state) {
        for (int k = 0; k < curNumValidMarkers; ++k) {
            const GenoBufItem& it = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(k)])];
            tile1_valid_extract_.push_back(it.extractedMarkerIndex);
            tile1_valid_mean_.push_back(it.mean);
            tile1_valid_has_miss_.push_back(it.nValidN < full_keep_ct ? uint8_t{1} : uint8_t{0});
        }
    }

    // Compact valid columns forward in-place for the FP32 path when any marker failed filters.
    if (!use_int8_grm && curNumValidMarkers < num_marker) {
        float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        for (int k = 0; k < curNumValidMarkers; ++k) {
            const int src_idx = validIndexBuf[static_cast<size_t>(k)];
            if (src_idx != k) {
                std::memcpy(std_geno_base + static_cast<size_t>(k) * stdGenoLD,
                            std_geno_base + static_cast<size_t>(src_idx) * stdGenoLD,
                            grm_bytes_std_geno);
            }
        }
    }
    if (!grm_skip_global_state) {
        for (int i = 0; i < curNumValidMarkers; i++) {
            sd.push_back(gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(i)])].sd);
        }
    }

    constexpr int kTile = kGrmStripRows;
    const int decode_rows = grm_tile_cols;  // == grm_tile_re (tiled) or grm_n (dense)
    const int num_strips = (grm_tile_rows + kTile - 1) / kTile;
    bool any_missing_full = false;
    bool tile_has_missing = false;

    for (int m = 0; m < curNumValidMarkers; ++m) {
        if (gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(m)])].nValidN < full_keep_ct) {
            any_missing_full = true;
            break;
        }
    }
    if (any_missing_full) {
        grm_any_missing_seen_ = true;
    }

    if (use_int8_grm) {
        int8_t* const kmajor_base = std::assume_aligned<64>(stdGeno_i8_kmajor_.get());
        const int ldK = std::max(64, (curNumValidMarkers + 63) / 64 * 64);

        double S_mu2 = 0.0;
        for (int m = 0; m < curNumValidMarkers; ++m) {
            const double mu_m = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(m)])].mean;
            mu_vec_[static_cast<size_t>(m)] = mu_m;
            S_mu2 += mu_m * mu_m;
        }

        geno->decodeBedI8KMajor(buf, validIndexBuf.data(), curNumValidMarkers, decode_rows,
                                kmajor_base, ldK, mu_vec_.data(), r_raw_.data());

        marker_miss_idx_.clear();
        if (any_missing_full) {
            miss_col_of_.assign(static_cast<size_t>(curNumValidMarkers), -1);
            std::vector<int> miss_markers;
            for (int m = 0; m < curNumValidMarkers; ++m) {
                if (gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(m)])].nValidN < full_keep_ct) {
                    miss_col_of_[static_cast<size_t>(m)] = static_cast<int>(miss_markers.size());
                    miss_markers.push_back(m);
                }
            }
            const int n_miss_cols = static_cast<int>(miss_markers.size());
            if (n_miss_cols > stdGeno_i8_cols_) {
                stdGeno_i8 = make_aligned_uninitialized<int8_t, 64>(
                    static_cast<size_t>(n_miss_cols) * static_cast<size_t>(stdGenoLD));
                if (!stdGeno_i8) {
                    LOGGER.e(0, "can't allocate enough memory for the int8 missing-genotype columns.");
                }
                stdGeno_i8_cols_ = n_miss_cols;
            }
            int8_t* const miss_cols = stdGeno_i8.get();
            #pragma omp parallel for schedule(dynamic, 4)
            for (int j = 0; j < n_miss_cols; ++j) {
                const int bi = validIndexBuf[static_cast<size_t>(miss_markers[static_cast<size_t>(j)])];
                GenoBufItem &item = gbufitems[static_cast<size_t>(bi)];
                item.stats_only = false;
                item.lazy_missing = false;
                item.geno_i8_ptr = miss_cols + static_cast<size_t>(j) * stdGenoLD;
                geno->getGenoDouble(buf, bi, &item);
                item.geno_i8_ptr = nullptr;
            }
            std::memcpy(r_tilde_.data(), r_raw_.data(), static_cast<size_t>(decode_rows) * sizeof(double));
            const int num_words = (decode_rows + 63) / 64;
            const uintptr_t last_word_mask = (decode_rows % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (decode_rows % 64)) - 1);
            marker_miss_offset_[0] = 0;
            for (int m = 0; m < curNumValidMarkers; ++m) {
                const GenoBufItem& item = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(m)])];
                if (item.nValidN < full_keep_ct && !item.missing.empty()) {
                    const double mu2 = mu_vec_[static_cast<size_t>(m)] * mu_vec_[static_cast<size_t>(m)];
                    const uintptr_t* miss_words = item.missing.data();
                    for (int w = 0; w < num_words; ++w) {
                        uintptr_t word = miss_words[w];
                        if (w == num_words - 1) word &= last_word_mask;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            const int s = w * 64 + bit;
                            r_tilde_[static_cast<size_t>(s)] += mu2;
                            marker_miss_idx_.push_back(s);
                            word &= (word - 1);
                        }
                    }
                }
                marker_miss_offset_[static_cast<size_t>(m + 1)] = static_cast<int>(marker_miss_idx_.size());
            }
            tile_has_missing = !marker_miss_idx_.empty();
            if (tile_has_missing) {
                std::fill(sample_miss_offset_.begin(), sample_miss_offset_.begin() + decode_rows + 1, 0);
                for (int s : marker_miss_idx_) {
                    sample_miss_offset_[static_cast<size_t>(s + 1)]++;
                }
                for (int s = 0; s < decode_rows; ++s) {
                    sample_miss_offset_[static_cast<size_t>(s + 1)] += sample_miss_offset_[static_cast<size_t>(s)];
                }
                sample_miss_m_.resize(static_cast<size_t>(sample_miss_offset_[static_cast<size_t>(decode_rows)]));
                std::vector<int> write_pos(sample_miss_offset_.begin(),
                                           sample_miss_offset_.begin() + decode_rows);
                for (int m = 0; m < curNumValidMarkers; ++m) {
                    const int mb = marker_miss_offset_[static_cast<size_t>(m)];
                    const int me = marker_miss_offset_[static_cast<size_t>(m + 1)];
                    for (int idx = mb; idx < me; ++idx) {
                        const int s = marker_miss_idx_[static_cast<size_t>(idx)];
                        sample_miss_m_[static_cast<size_t>(write_pos[static_cast<size_t>(s)]++)] = m;
                    }
                }
            }
        }

        if (grm_is_i32_ &&
            (tile_has_missing ||
             static_cast<uint64_t>(grm_tile_i32_markers_) + static_cast<uint64_t>(curNumValidMarkers) > kMaxInt32Markers)) {
            promote_i32_to_f64();
        }

        const bool first_accum = grm_first_accum;

        if (grm_is_i32_) {
            const int32_t beta = first_accum ? 0 : 1;
            constexpr int kDiagSub = 1024;
            if (grm_i32_diag_scratch_.size() < static_cast<size_t>(kDiagSub) * kDiagSub) {
                grm_i32_diag_scratch_.resize(static_cast<size_t>(kDiagSub) * kDiagSub);
            }

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * kTile;
                const int rows = std::min(kTile, grm_tile_rows - r0);
                const int global_r0 = grm_tile_rs + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;
                int32_t* const strip_base = std::assume_aligned<64>(grm_i32_strips_[static_cast<size_t>(s)].get());

                if (global_r0 > 0) {
                    const int8_t* A_reord_ptr = nullptr;
                    if (global_r0 > kTile) {
                        const size_t rbytes = gcta_u8s8s32_reorder_bt_size(rows, curNumValidMarkers);
                        if (rbytes > 0) {
                            if (i8_reordered_B_.size() < rbytes) i8_reordered_B_.resize(rbytes);
                            gcta_u8s8s32_reorder_bt(rows, curNumValidMarkers, A_strip, ldK, i8_reordered_B_.data());
                            A_reord_ptr = i8_reordered_B_.data();
                        }
                    }
                    for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                        const int cols = std::min(kTile, global_r0 - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        gcta_gemm_u8s8s32_kmajor_nt_beta_reordered(
                            rows, cols, curNumValidMarkers,
                            A_reord_ptr, A_strip, ldK,
                            B_strip,
                            strip_base + static_cast<size_t>(c0) * rows,
                            rows, beta,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);
                    }
                }

                // Diagonal rows x rows lower triangle: sub-tile by kDiagSub = 1024
                for (int d0 = 0; d0 < rows; d0 += kDiagSub) {
                    const int db = std::min(kDiagSub, rows - d0);
                    const int8_t* const A_d = kmajor_base + static_cast<size_t>(global_r0 + d0) * ldK;
                    gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                        db, curNumValidMarkers, A_d, ldK,
                        grm_i32_diag_scratch_.data(), db, 0,
                        i8_scratch_A_f32_, i8_scratch_C_f32_);

                    const int32_t* const diag_src = grm_i32_diag_scratch_.data();
                    int32_t* const diag_dst = strip_base + d0
                                              + static_cast<size_t>(global_r0 + d0) * rows;
                    #pragma omp parallel if (db >= 256)
                    {
                        const int tid = omp_get_thread_num();
                        const int nth = omp_get_num_threads();
                        const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(db, tid, nth);
                        for (int c = c_beg; c < c_end; ++c) {
                            const int32_t* __restrict__ src_col = diag_src + c + static_cast<size_t>(c) * db;
                            int32_t* __restrict__ dst_col       = diag_dst + c + static_cast<size_t>(c) * rows;
                            const int len = db - c;
                            if (first_accum) {
                                std::memcpy(dst_col, src_col, static_cast<size_t>(len) * sizeof(int32_t));
                            } else {
                                #pragma omp simd
                                for (int r = 0; r < len; ++r) {
                                    dst_col[r] += src_col[r];
                                }
                            }
                        }
                    }

                    const int rem = rows - (d0 + db);
                    if (rem > 0) {
                        gcta_gemm_u8s8s32_kmajor_nt_beta(
                            rem, db, curNumValidMarkers,
                            kmajor_base + static_cast<size_t>(global_r0 + d0 + db) * ldK, ldK,
                            A_d,
                            strip_base + (d0 + db) + static_cast<size_t>(global_r0 + d0) * rows,
                            rows, beta,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);
                    }
                }
            }

            S_total_ += S_mu2;
            double* __restrict__ R_tot = R_total_.data();
            const double* __restrict__ r_cur = r_raw_.data();
            #pragma omp parallel for schedule(static) if (decode_rows >= 4096)
            for (int i = 0; i < decode_rows; ++i) {
                R_tot[i] += r_cur[i];
            }
            grm_tile_i32_markers_ += static_cast<uint32_t>(curNumValidMarkers);
        } else {
            const int max_tile_dim = std::min(kTile, std::max(1, grm_n));
            const size_t need_tile = static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim);
            if (grm_i32_tile_.size() < need_tile) {
                grm_i32_tile_.resize(need_tile);
            }

            const bool block_has_missing = tile_has_missing;
            const double* const r_row_vec = block_has_missing ? r_tilde_.data() : r_raw_.data();

            std::vector<int> row_miss_b;
            std::vector<int> row_miss_e;
            std::vector<int> active_row_miss_markers;
            if (block_has_missing) {
                row_miss_b.resize(static_cast<size_t>(curNumValidMarkers));
                row_miss_e.resize(static_cast<size_t>(curNumValidMarkers));
                active_row_miss_markers.reserve(static_cast<size_t>(curNumValidMarkers));
            }

            auto prepare_row_missing = [&](int global_r0, int rows) {
                if (!block_has_missing) return;
                #pragma omp parallel for schedule(static) if (curNumValidMarkers >= 256)
                for (int m = 0; m < curNumValidMarkers; ++m) {
                    const int mb = marker_miss_offset_[static_cast<size_t>(m)];
                    const int me = marker_miss_offset_[static_cast<size_t>(m + 1)];
                    if (mb == me) {
                        row_miss_b[static_cast<size_t>(m)] = row_miss_e[static_cast<size_t>(m)] = mb;
                        continue;
                    }
                    const int b = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + mb,
                        marker_miss_idx_.begin() + me,
                        global_r0) - marker_miss_idx_.begin());
                    const int e = static_cast<int>(std::lower_bound(
                        marker_miss_idx_.begin() + b,
                        marker_miss_idx_.begin() + me,
                        global_r0 + rows) - marker_miss_idx_.begin());
                    row_miss_b[static_cast<size_t>(m)] = b;
                    row_miss_e[static_cast<size_t>(m)] = e;
                }
                active_row_miss_markers.clear();
                for (int m = 0; m < curNumValidMarkers; ++m) {
                    if (row_miss_b[static_cast<size_t>(m)] < row_miss_e[static_cast<size_t>(m)]) {
                        active_row_miss_markers.push_back(m);
                    }
                }
            };

            auto accum_col_with_centering = [&](int j, int row_start, int len,
                                                const int32_t* __restrict__ src_col,
                                                double* __restrict__ dst_col) {
                const double r_j_plus_S = S_mu2 - r_raw_[static_cast<size_t>(j)];
                const double* __restrict__ r_row_slice = r_row_vec + row_start;
                if (first_accum) {
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] = static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                    }
                } else {
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] += static_cast<double>(src_col[r]) - r_row_slice[r] + r_j_plus_S;
                    }
                }
                if (!block_has_missing) return;

                const int cm_b = sample_miss_offset_[static_cast<size_t>(j)];
                const int cm_e = sample_miss_offset_[static_cast<size_t>(j + 1)];
                for (int idx = cm_b; idx < cm_e; ++idx) {
                    const int m = sample_miss_m_[static_cast<size_t>(idx)];
                    const double mu_m = mu_vec_[static_cast<size_t>(m)];
                    const double neg_mu2 = -mu_m * mu_m;
                    const int8_t* __restrict__ col_m = stdGeno_i8.get()
                                          + static_cast<size_t>(miss_col_of_[static_cast<size_t>(m)]) * stdGenoLD + row_start;
                    #pragma omp simd
                    for (int r = 0; r < len; ++r) {
                        dst_col[r] += static_cast<double>(col_m[r]) * mu_m + neg_mu2;
                    }
                    for (int r_idx = row_miss_b[static_cast<size_t>(m)]; r_idx < row_miss_e[static_cast<size_t>(m)]; ++r_idx) {
                        const int s = marker_miss_idx_[static_cast<size_t>(r_idx)];
                        if (s >= row_start) {
                            dst_col[s - row_start] -= neg_mu2;
                        }
                    }
                }

                const int8_t* __restrict__ row_j_kmajor = kmajor_base + static_cast<size_t>(j) * ldK;
                for (int m : active_row_miss_markers) {
                    const int8_t g_jm = row_j_kmajor[m];
                    if (g_jm != 0) {
                        const double corr = mu_vec_[static_cast<size_t>(m)] * static_cast<double>(g_jm);
                        for (int r_idx = row_miss_b[static_cast<size_t>(m)]; r_idx < row_miss_e[static_cast<size_t>(m)]; ++r_idx) {
                            const int s = marker_miss_idx_[static_cast<size_t>(r_idx)];
                            if (s >= row_start) {
                                dst_col[s - row_start] += corr;
                            }
                        }
                    }
                }
            };

            for (int s = 0; s < num_strips; ++s) {
                const int r0 = s * kTile;
                const int rows = std::min(kTile, grm_tile_rows - r0);
                const int global_r0 = grm_tile_rs + r0;
                const int8_t* const A_strip = kmajor_base + static_cast<size_t>(global_r0) * ldK;
                double* const strip_base = std::assume_aligned<64>(grm_strips_[static_cast<size_t>(s)].get());

                prepare_row_missing(global_r0, rows);

                if (global_r0 > 0) {
                    const int8_t* A_reord_ptr = nullptr;
                    if (global_r0 > kTile) {
                        const size_t rbytes = gcta_u8s8s32_reorder_bt_size(rows, curNumValidMarkers);
                        if (rbytes > 0) {
                            if (i8_reordered_B_.size() < rbytes) i8_reordered_B_.resize(rbytes);
                            gcta_u8s8s32_reorder_bt(rows, curNumValidMarkers, A_strip, ldK, i8_reordered_B_.data());
                            A_reord_ptr = i8_reordered_B_.data();
                        }
                    }
                    for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                        const int cols = std::min(kTile, global_r0 - c0);
                        const int8_t* const B_strip = kmajor_base + static_cast<size_t>(c0) * ldK;
                        gcta_gemm_u8s8s32_kmajor_nt_beta_reordered(
                            rows, cols, curNumValidMarkers,
                            A_reord_ptr, A_strip, ldK,
                            B_strip,
                            grm_i32_tile_.data(), rows, 0,
                            i8_scratch_A_f32_, i8_scratch_B_f32_, i8_scratch_C_f32_);

                        #pragma omp parallel for schedule(static)
                        for (int c = 0; c < cols; ++c) {
                            double* dst_col = strip_base + static_cast<size_t>(c0 + c) * rows;
                            const int32_t* src_col = grm_i32_tile_.data() + static_cast<size_t>(c) * rows;
                            accum_col_with_centering(c0 + c, global_r0, rows, src_col, dst_col);
                        }
                    }
                }

                gcta_syrk_u8s8s32_lower_kmajor_nt_beta(
                    rows, curNumValidMarkers,
                    A_strip, ldK,
                    grm_i32_tile_.data(), rows, 0,
                    i8_scratch_A_f32_, i8_scratch_C_f32_);

                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    const int nth = omp_get_num_threads();
                    const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(rows, tid, nth);
                    for (int c = c_beg; c < c_end; ++c) {
                        double* dst_col = strip_base + c + static_cast<size_t>(global_r0 + c) * rows;
                        const int32_t* src_col = grm_i32_tile_.data() + c + static_cast<size_t>(c) * rows;
                        const int j = global_r0 + c;
                        accum_col_with_centering(j, j, rows - c, src_col, dst_col);
                    }
                }
            }
        }
    } else {
        const float* const std_geno_base = std::assume_aligned<64>(stdGeno.get());
        const bool first_accum = grm_first_accum;

        for (int s = 0; s < num_strips; ++s) {
            const int r0 = s * kTile;
            const int rows = std::min(kTile, grm_tile_rows - r0);
            const int global_r0 = grm_tile_rs + r0;
            const float* const A_strip = std_geno_base + global_r0;
            double* const strip_base = std::assume_aligned<64>(grm_strips_[static_cast<size_t>(s)].get());

            for (int c0 = 0; c0 < global_r0; c0 += kTile) {
                const int cols = std::min(kTile, global_r0 - c0);
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            rows, cols, curNumValidMarkers,
                            1.0f,
                            A_strip, stdGenoLD,
                            std_geno_base + c0, stdGenoLD,
                            0.0f,
                            grm_f32_tile_buf_.data(), rows);

                #pragma omp parallel for schedule(static)
                for (int c = 0; c < cols; ++c) {
                    double* __restrict__ dst_col = strip_base + static_cast<size_t>(c0 + c) * rows;
                    const float* __restrict__ src_col = grm_f32_tile_buf_.data() + static_cast<size_t>(c) * rows;
                    if (first_accum) {
                        #pragma omp simd
                        for (int r = 0; r < rows; ++r) dst_col[r] = static_cast<double>(src_col[r]);
                    } else {
                        #pragma omp simd
                        for (int r = 0; r < rows; ++r) dst_col[r] += static_cast<double>(src_col[r]);
                    }
                }
            }

            cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        rows, curNumValidMarkers,
                        1.0f,
                        A_strip, stdGenoLD,
                        0.0f,
                        grm_f32_tile_buf_.data(), rows);

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                const auto [c_beg, c_end] = gcta_grm_detail::lower_tri_col_range(rows, tid, nth);
                for (int c = c_beg; c < c_end; ++c) {
                    double* __restrict__ dst_col = strip_base + c + static_cast<size_t>(global_r0 + c) * rows;
                    const float* __restrict__ src_col = grm_f32_tile_buf_.data() + c + static_cast<size_t>(c) * rows;
                    const int len = rows - c;
                    if (first_accum) {
                        #pragma omp simd
                        for (int r = 0; r < len; ++r) dst_col[r] = static_cast<double>(src_col[r]);
                    } else {
                        #pragma omp simd
                        for (int r = 0; r < len; ++r) dst_col[r] += static_cast<double>(src_col[r]);
                    }
                }
            }
        }

        if (any_missing_full) {
            const int num_words = (decode_rows + 63) / 64;
            const uintptr_t last_word_mask = (decode_rows % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (decode_rows % 64)) - 1);
            for (int m = 0; m < curNumValidMarkers && !tile_has_missing; ++m) {
                const GenoBufItem& it = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(m)])];
                if (it.nValidN < full_keep_ct && !it.missing.empty()) {
                    for (int w = 0; w < num_words; ++w) {
                        uintptr_t word = it.missing[static_cast<size_t>(w)];
                        if (w == num_words - 1) word &= last_word_mask;
                        if (word != 0) {
                            tile_has_missing = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    grm_first_accum = false;

    if (!any_missing_full) {
        if (!grm_skip_global_state) {
            finished_marker += num_marker;
            numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
        }
        return;
    }

    if (!tile_has_missing) {
        if (!grm_skip_global_state) {
            const int num_words_all = (grm_n + 63) / 64;
            const uintptr_t last_mask_all = (grm_n % 64 == 0)
                ? ~static_cast<uintptr_t>(0)
                : ((static_cast<uintptr_t>(1) << (grm_n % 64)) - 1);
            #pragma omp parallel for schedule(static) if (num_words_all >= 64)
            for (int w = 0; w < num_words_all; ++w) {
                const int base_s = w * 64;
                for (int k = 0; k < curNumValidMarkers; ++k) {
                    const GenoBufItem& it = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(k)])];
                    if (it.nValidN < full_keep_ct && !it.missing.empty()) {
                        uintptr_t word = it.missing[static_cast<size_t>(w)];
                        if (w == num_words_all - 1) word &= last_mask_all;
                        while (word != 0) {
                            const int bit = std::countr_zero(word);
                            sub_miss[static_cast<size_t>(base_s + bit)]++;
                            word &= (word - 1);
                        }
                    }
                }
            }
            finished_marker += num_marker;
            numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
        }
        return;
    }

    ensure_N_allocated();

    const int markerPerN = static_cast<int>(sizeof(uintptr_t) * CHAR_BIT);
    const int numNblock = (curNumValidMarkers + markerPerN - 1) / markerPerN;
    const int active_samples = (!grm_skip_global_state) ? grm_n : decode_rows;
    const int numNSampleBlock = (active_samples + markerPerN - 1) / markerPerN;
    const size_t blockStride = static_cast<size_t>(numNSampleBlock) * static_cast<size_t>(markerPerN);
    const size_t need_miss_words = blockStride * static_cast<size_t>(numNblock);
    if (sampleMissBuf.size() < need_miss_words) {
        sampleMissBuf.resize(need_miss_words);
    }

    #pragma omp parallel
    {
        std::array<uintptr_t, 64> transposeInput{};
        alignas(32) plink2::VecW transposeScratch[plink2::kPglBitTransposeBufvecs];

        #pragma omp for schedule(static)
        for (int j = 0; j < numNSampleBlock; j++) {
            const int baseMissIndex = j * markerPerN;
            for (int i = 0; i < numNblock; i++) {
                const int baseMarkerIndex = markerPerN * i;
                const int lastValidIndex  = std::min(markerPerN * (i + 1), curNumValidMarkers);
                uintptr_t *block_buf = sampleMissBuf.data() + static_cast<size_t>(i) * blockStride;
                if (markerPerN == 64) {
                    for (int k = baseMarkerIndex; k < lastValidIndex; k++) {
                        const GenoBufItem &it = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(k)])];
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] =
                            (it.nValidN < full_keep_ct && !it.missing.empty()) ? it.missing[static_cast<size_t>(j)] : 0UL;
                    }
                    for (int k = lastValidIndex; k < markerPerN * (i + 1); k++) {
                        transposeInput[static_cast<std::size_t>(k - baseMarkerIndex)] = 0UL;
                    }
                    plink2::TransposeBitblock(
                        transposeInput.data(), 1, 1,
                        static_cast<uint32_t>(markerPerN), static_cast<uint32_t>(markerPerN),
                        &block_buf[baseMissIndex], transposeScratch);
                } else {
                    for (int k = baseMarkerIndex; k < lastValidIndex; k++) {
                        const GenoBufItem &it = gbufitems[static_cast<size_t>(validIndexBuf[static_cast<size_t>(k)])];
                        block_buf[baseMissIndex + k - baseMarkerIndex] =
                            (it.nValidN < full_keep_ct && !it.missing.empty()) ? it.missing[static_cast<size_t>(j)] : 0UL;
                    }
                    for (int k = lastValidIndex; k < markerPerN * (i + 1); k++) {
                        block_buf[baseMissIndex + k - baseMarkerIndex] = 0UL;
                    }
                }
                if (!grm_skip_global_state) {
                    for (int k = baseMissIndex; k < baseMissIndex + markerPerN; k++) {
                        sub_miss[static_cast<size_t>(k)] += std::popcount(block_buf[k]);
                    }
                }
            }
        }

        #pragma omp for schedule(static)
        for (int index = 0; index < static_cast<int>(index_grm_pairs.size()); index++) {
            const auto index_pair = index_grm_pairs[static_cast<size_t>(index)];
            for (int i = 0; i < numNblock; i++) {
                N_thread(index_pair.first, index_pair.second,
                         sampleMissBuf.data() + static_cast<size_t>(i) * blockStride);
            }
        }
    }

    if (!grm_skip_global_state) {
        finished_marker += num_marker;
        numValidMarkers += static_cast<uint32_t>(curNumValidMarkers);
    }
}

void GRM::deduce_GRM(){
    LOGGER.i(0, "The GRM computation is completed.");
    float thresh = -99.0f;
    bool isSparse = false;
    if (sparse_cutoff.has_value()) {
        thresh = *sparse_cutoff;
        isSparse = true;
        LOGGER.i(0, "Saving sparse GRM with a cutoff " + to_string(thresh) + "...");
    } else if (write_fp16_) {
        LOGGER.i(0, "Saving native FP16 GRM...");
    } else {
        LOGGER.i(0, "Saving GRM...");
    }

    const std::string grm_path = isSparse ? (o_name + ".grm.sp") : (o_name + (write_fp16_ ? ".grm.f16" : ".grm.bin"));
    const std::string N_path   = o_name + ".grm.N.bin";
    const bool need_N_file = !isSparse && (!write_fp16_ || grm_any_missing_seen_);

    UniqueFile sparse_out;
    std::unique_ptr<gcta_io::AsyncSeqWriter> grm_writer;
    gcta_grm_detail::GrmUniqueFd N_fd;

    const uint64_t n_out = static_cast<uint64_t>(num_individual);
    const uint64_t body_off = write_fp16_ ? gcta_grm_io::grm_f16_body_offset(n_out) : 0ULL;

    if (isSparse) {
        sparse_out = open_file(grm_path, "wb");
        if (!sparse_out) LOGGER.error(0, "can't open {} to write", grm_path);
        setvbuf(sparse_out.get(), nullptr, _IOFBF, 4 << 20);
    } else {
        try {
            grm_writer = std::make_unique<gcta_io::AsyncSeqWriter>(
                grm_path, gcta_grm_detail::kGrmAsyncSlotBytes, static_cast<off_t>(body_off));
            if (write_fp16_) {
                gcta_grm_io::write_grm_f16_draft_prefix(*grm_writer, n_out);
                fp16_diag_.assign(static_cast<size_t>(n_out), 0.0f);
                fp16_trace_K_ = 0.0;
                fp16_trace_K2_ = 0.0;
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("deduce_GRM: ") + e.what());
        }
        if (need_N_file) {
            N_fd = gcta_grm_detail::open_posix_write_trunc(N_path);
            if (!N_fd) LOGGER.error(0, "can't open {} to write", N_path);
        }
    }

    double mtd_weight = 1.0;
    if (isMtd) {
        double weight = 0.0;
        if (!isDominance) {
            for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
        } else {
            for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
        }
        if (weight > 0.0) mtd_weight = static_cast<double>(numValidMarkers) / weight;
    }

    grm_tile_rs   = static_cast<int>(part_keep_indices.first);
    grm_tile_re   = static_cast<int>(part_keep_indices.second) + 1;
    grm_tile_rows = grm_m;
    grm_tile_cols = grm_n;

    std::vector<float> const_N_buf;
    std::string sparse_buf;
    flush_grm_tile(grm_writer.get(), N_fd.get(), sparse_out.get(), grm_path, N_path,
                   thresh, isSparse, mtd_weight, const_N_buf, sparse_buf);

    if (!isSparse) {
        try {
            if (write_fp16_) {
                gcta_grm_io::GrmFp16Header final_hdr{};
                std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
                final_hdr.version = gcta_grm_io::kGrmF16Version;
                final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
                final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
                final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized |
                                  gcta_grm_io::kGrmF16FlagHasTraces |
                                  gcta_grm_io::kGrmF16FlagHasDiagF32 |
                                  (!grm_any_missing_seen_ ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u) |
                                  (isMtd ? gcta_grm_io::kGrmF16FlagVanRaden : 0u) |
                                  (isDominance ? gcta_grm_io::kGrmF16FlagDominance : 0u) |
                                  (is_homogametic_ ? gcta_grm_io::kGrmF16FlagHomogametic : 0u);
                final_hdr.n = n_out;
                final_hdr.valid_markers = numValidMarkers;
                if (!grm_any_missing_seen_) {
                    final_hdr.m_snps_mean = static_cast<double>(numValidMarkers);
                } else {
                    double miss_sum = 0.0;
                    for (uint64_t i = 0; i < n_out; ++i) miss_sum += static_cast<double>(sub_miss[static_cast<size_t>(i)]);
                    final_hdr.m_snps_mean = (n_out > 0)
                        ? (static_cast<double>(numValidMarkers) - miss_sum / static_cast<double>(n_out))
                        : 0.0;
                }
                final_hdr.id_hash = grm_id_hash_;
                final_hdr.trace_K = fp16_trace_K_;
                final_hdr.trace_K2 = fp16_trace_K2_;
                final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
                final_hdr.body_offset = body_off;
                final_hdr.part_row_start = 0;
                final_hdr.part_row_end = n_out;
                final_hdr.mtd_weight = mtd_weight;
                gcta_grm_io::finalize_grm_f16_file(*grm_writer, final_hdr, fp16_diag_.data(), n_out);
            } else {
                grm_writer->finish();
            }
        } catch (const std::exception& e) {
            LOGGER.e(0, std::string("Failed finalizing [") + grm_path + "]: " + e.what());
        }
    }
    grm_writer.reset();
    N_fd.reset();
    sparse_out.reset();

    if (isSparse) {
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
    } else if (write_fp16_) {
        LOGGER.i(0, "Native FP16 GRM has been saved in the file [" + grm_path + "]" +
                    (!grm_any_missing_seen_
                         ? " (zero missing genotypes; .grm.N.bin omitted, m_snps=" + to_string(numValidMarkers) + " in header)"
                         : " (missingness present; per-pair counts in [" + N_path + "])"));
    } else {
        LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
        LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
    }
}

// ---------------------------------------------------------------------------
// GRM::flush_grm_tile — single-stream AsyncSeqWriter flush from 4096-row strips.
// Zero concurrent 2-file writes: streams GRM strips asynchronously through
// grm_writer, then (if N_fd >= 0) drains grm_writer and writes N sequentially.
// ---------------------------------------------------------------------------
void GRM::flush_grm_tile(gcta_io::AsyncSeqWriter* grm_writer,
                         int N_fd,
                         FILE *sparse_out,
                         const std::string& grm_path,
                         const std::string& N_path,
                         float thresh,
                         bool isSparse,
                         double mtd_weight,
                         std::vector<float>& const_N_buf,
                         std::string& sparse_buf){
    const int tile_rs   = grm_tile_rs;
    const int tile_re   = grm_tile_re;
    const int tile_rows = grm_tile_rows;
    const int tile_cols = grm_tile_cols;

    constexpr size_t kConstNFloats = (64ULL << 20) / sizeof(float); // 64 MiB constant buffer

    // Edge case: 0 valid markers accumulated (all-missing / all-filtered run).
    if (grm_first_accum || numValidMarkers == 0) {
        if (!isSparse) {
            const uint64_t dre = static_cast<uint64_t>(tile_re);
            const uint64_t drs = static_cast<uint64_t>(tile_rs);
            const size_t total_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);
            const size_t elem_bytes = write_fp16_ ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);
            const size_t total_grm_bytes = total_elems * elem_bytes;
            try {
                size_t rem = total_grm_bytes;
                while (rem > 0) {
                    AlignedRawBuffer<char, 64>& slot = grm_writer->acquire_slot();
                    const size_t c = std::min(rem, slot.size());
                    std::memset(slot.data(), 0, c);
                    grm_writer->submit_slot(c);
                    rem -= c;
                }
                if (N_fd >= 0) {
                    grm_writer->drain();
                    const size_t total_n_bytes = total_elems * sizeof(float);
                    const size_t buf_floats = std::min(total_elems, kConstNFloats);
                    if (const_N_buf.size() < buf_floats) const_N_buf.resize(buf_floats);
                    std::fill(const_N_buf.begin(), const_N_buf.begin() + static_cast<std::ptrdiff_t>(buf_floats), 0.0f);
                    size_t n_rem = total_n_bytes;
                    const size_t chunk_cap = buf_floats * sizeof(float);
                    while (n_rem > 0) {
                        const size_t c = std::min(n_rem, chunk_cap);
                        gcta_io::write_exact_throwing(N_fd, const_N_buf.data(), c, N_path);
                        n_rem -= c;
                    }
                }
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("flush_grm_tile zero-marker write failed: ") + e.what());
            }
        } else if (0.0f >= thresh) {
            for (int pair1 = tile_rs; pair1 < tile_re; ++pair1) {
                sparse_buf.clear();
                for (int i = 0; i <= pair1; ++i) {
                    std::format_to(std::back_inserter(sparse_buf), "{}\t{}\t0\n", pair1, i);
                }
                if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                    LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                }
            }
        }
        return;
    }

    const bool zero_missing = (N == nullptr);
    const double inv_scale = mtd_weight / static_cast<double>(numValidMarkers);
    const size_t out_elem_bytes = write_fp16_ ? sizeof(gcta_grm_io::GrmFp16) : sizeof(float);

    // Target <= 64 MiB packed GRM per AsyncSeqWriter slot (bounded even at N=1,000,000).
    const size_t kTargetStripElems = gcta_grm_detail::kGrmAsyncSlotBytes / out_elem_bytes;
    const size_t raw_step = kTargetStripElems / std::max<size_t>(1, static_cast<size_t>(tile_cols));
    const int sub_step_rows = static_cast<int>(std::clamp<size_t>((raw_step / 64ULL) * 64ULL, 64ULL, 1024ULL));

    const uint32_t* const N_base       = N ? std::assume_aligned<64>(N.get()) : nullptr;
    const uint32_t* const sub_miss_ptr = sub_miss.data();
    const double* const R_ptr          = grm_is_i32_ ? R_total_.data() : nullptr;
    const double S_tot                 = S_total_;

    // Unified packed triangular row lookup for N (identical in tiled and dense paths).
    const auto get_N_row = [&](int pair1) -> const uint32_t* {
        const size_t start_pos = static_cast<size_t>(pair1 + 1 + tile_rs) *
                                 static_cast<size_t>(pair1 - tile_rs) / 2;
        return N_base + start_pos;
    };

    std::vector<size_t> row_off(static_cast<size_t>(sub_step_rows) + 1);
    AlignedRawBuffer<float, 64> sparse_packed_grm; // used only when isSparse == true

    const int num_strips = (tile_rows + kGrmStripRows - 1) / kGrmStripRows;

    try {
        for (int s = 0; s < num_strips; ++s) {
            const int strip_local_r0 = s * kGrmStripRows;
            const int strip_h        = std::min(kGrmStripRows, tile_rows - strip_local_r0);
            const int strip_global_r0 = tile_rs + strip_local_r0;
            const int strip_global_r1 = strip_global_r0 + strip_h;

            const double* const strip_f64 = (!grm_is_i32_ && !grm_strips_.empty())
                ? std::assume_aligned<64>(grm_strips_[static_cast<size_t>(s)].get()) : nullptr;
            const int32_t* const strip_i32 = (grm_is_i32_ && !grm_i32_strips_.empty())
                ? std::assume_aligned<64>(grm_i32_strips_[static_cast<size_t>(s)].get()) : nullptr;

            for (int block_start = strip_global_r0; block_start < strip_global_r1; block_start += sub_step_rows) {
                const int block_end    = std::min(block_start + sub_step_rows, strip_global_r1);
                const int block_rows   = block_end - block_start;
                const int strip_row_r0 = block_start - strip_global_r0;

                size_t total_packed = 0;
                for (int r = 0; r < block_rows; ++r) {
                    row_off[static_cast<size_t>(r)] = total_packed;
                    total_packed += static_cast<size_t>(block_start + r + 1);
                }
                row_off[static_cast<size_t>(block_rows)] = total_packed;

                const size_t strip_bytes = total_packed * out_elem_bytes;
                char* out_raw = nullptr;
                if (!isSparse) {
                    AlignedRawBuffer<char, 64>& slot = grm_writer->acquire_slot();
                    if (slot.size() < strip_bytes) slot.resize(strip_bytes);
                    out_raw = slot.data();
                } else {
                    if (sparse_packed_grm.size() < total_packed) sparse_packed_grm.resize(total_packed);
                    out_raw = reinterpret_cast<char*>(sparse_packed_grm.data());
                }

                float* const out_f32 = (!write_fp16_ || isSparse) ? reinterpret_cast<float*>(out_raw) : nullptr;
                gcta_grm_io::GrmFp16* const out_f16 = (write_fp16_ && !isSparse)
                    ? reinterpret_cast<gcta_grm_io::GrmFp16*>(out_raw) : nullptr;

                double tr_K_strip = 0.0;
                double tr_K2_strip = 0.0;

                if (zero_missing && grm_is_i32_) {
                    constexpr int kBlk = 64;
                    const int n_cblk = (block_end + kBlk - 1) / kBlk;
                    #pragma omp parallel for reduction(+:tr_K_strip, tr_K2_strip) schedule(static)
                    for (int cb = 0; cb < n_cblk; ++cb) {
                        const int c0 = cb * kBlk;
                        const int c_len = std::min(kBlk, block_end - c0);
                        alignas(64) float tmp[kBlk][kBlk];
                        for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                            const int r_len = std::min(kBlk, block_rows - r0);
                            const int global_r0 = block_start + r0;
                            if (c0 >= global_r0 + r_len) continue;
                            const int local_r0 = strip_row_r0 + r0;
                            const double* __restrict__ R_row = R_ptr + global_r0;
                            for (int dc = 0; dc < c_len; ++dc) {
                                const int c = c0 + dc;
                                const int dr_start = std::max(0, c - global_r0);
                                if (dr_start >= r_len) break;
                                const int32_t* __restrict__ col_src =
                                    strip_i32 + static_cast<size_t>(c) * strip_h + local_r0;
                                const double r_j_plus_S = S_tot - R_ptr[c];
                                for (int dr = dr_start; dr < r_len; ++dr) {
                                    tmp[dr][dc] = static_cast<float>(
                                        (static_cast<double>(col_src[dr]) - R_row[dr] + r_j_plus_S) * inv_scale);
                                }
                            }
                            if (out_f16) {
                                for (int dr = 0; dr < r_len; ++dr) {
                                    const int pair1 = global_r0 + dr;
                                    const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                    if (valid_c <= 0) continue;
                                    gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + row_off[static_cast<size_t>(r0 + dr)] + c0;
                                    const float* __restrict__ src = tmp[dr];
                                    const bool has_diag = (c0 + valid_c - 1 == pair1);
                                    const int off_end = has_diag ? (valid_c - 1) : valid_c;
                                    double off_sq = 0.0;
                                    #pragma omp simd reduction(+:off_sq)
                                    for (int dc = 0; dc < off_end; ++dc) {
                                        const float v = src[dc];
                                        dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        off_sq += vd * vd;
                                    }
                                    tr_K2_strip += 2.0 * off_sq;
                                    if (has_diag) {
                                        const float dv = src[off_end];
                                        dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                        fp16_diag_[static_cast<size_t>(pair1)] = dv;
                                        const double dd = static_cast<double>(dv);
                                        tr_K_strip += dd;
                                        tr_K2_strip += dd * dd;
                                    }
                                }
                            } else {
                                for (int dr = 0; dr < r_len; ++dr) {
                                    const int pair1 = global_r0 + dr;
                                    const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                    if (valid_c <= 0) continue;
                                    std::memcpy(out_f32 + row_off[static_cast<size_t>(r0 + dr)] + c0,
                                                tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                                }
                            }
                        }
                    }
                } else if (zero_missing) {
                    constexpr int kBlk = 64;
                    const int n_cblk = (block_end + kBlk - 1) / kBlk;
                    #pragma omp parallel for reduction(+:tr_K_strip, tr_K2_strip) schedule(static)
                    for (int cb = 0; cb < n_cblk; ++cb) {
                        const int c0 = cb * kBlk;
                        const int c_len = std::min(kBlk, block_end - c0);
                        alignas(64) float tmp[kBlk][kBlk];
                        for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                            const int r_len = std::min(kBlk, block_rows - r0);
                            const int global_r0 = block_start + r0;
                            if (c0 >= global_r0 + r_len) continue;
                            const int local_r0 = strip_row_r0 + r0;
                            for (int dc = 0; dc < c_len; ++dc) {
                                const int c = c0 + dc;
                                const int dr_start = std::max(0, c - global_r0);
                                if (dr_start >= r_len) break;
                                const double* __restrict__ col_src =
                                    strip_f64 + static_cast<size_t>(c) * strip_h + local_r0;
                                for (int dr = dr_start; dr < r_len; ++dr) {
                                    tmp[dr][dc] = static_cast<float>(col_src[dr] * inv_scale);
                                }
                            }
                            if (out_f16) {
                                for (int dr = 0; dr < r_len; ++dr) {
                                    const int pair1 = global_r0 + dr;
                                    const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                    if (valid_c <= 0) continue;
                                    gcta_grm_io::GrmFp16* __restrict__ dst = out_f16 + row_off[static_cast<size_t>(r0 + dr)] + c0;
                                    const float* __restrict__ src = tmp[dr];
                                    const bool has_diag = (c0 + valid_c - 1 == pair1);
                                    const int off_end = has_diag ? (valid_c - 1) : valid_c;
                                    double off_sq = 0.0;
                                    #pragma omp simd reduction(+:off_sq)
                                    for (int dc = 0; dc < off_end; ++dc) {
                                        const float v = src[dc];
                                        dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        off_sq += vd * vd;
                                    }
                                    tr_K2_strip += 2.0 * off_sq;
                                    if (has_diag) {
                                        const float dv = src[off_end];
                                        dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                        fp16_diag_[static_cast<size_t>(pair1)] = dv;
                                        const double dd = static_cast<double>(dv);
                                        tr_K_strip += dd;
                                        tr_K2_strip += dd * dd;
                                    }
                                }
                            } else {
                                for (int dr = 0; dr < r_len; ++dr) {
                                    const int pair1 = global_r0 + dr;
                                    const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                    if (valid_c <= 0) continue;
                                    std::memcpy(out_f32 + row_off[static_cast<size_t>(r0 + dr)] + c0,
                                                tmp[dr], static_cast<size_t>(valid_c) * sizeof(float));
                                }
                            }
                        }
                    }
                } else {
                    constexpr int kBlk = 32;
                    const int n_cblk = (block_end + kBlk - 1) / kBlk;
                    const uint32_t valid_m = numValidMarkers;
                    #pragma omp parallel for reduction(+:tr_K_strip, tr_K2_strip) schedule(static)
                    for (int cb = 0; cb < n_cblk; ++cb) {
                        const int c0 = cb * kBlk;
                        const int c_len = std::min(kBlk, block_end - c0);
                        alignas(64) double tmp_d[kBlk][kBlk];
                        for (int r0 = 0; r0 < block_rows; r0 += kBlk) {
                            const int r_len = std::min(kBlk, block_rows - r0);
                            const int global_r0 = block_start + r0;
                            if (c0 >= global_r0 + r_len) continue;
                            const int local_r0 = strip_row_r0 + r0;
                            for (int dc = 0; dc < c_len; ++dc) {
                                const int c = c0 + dc;
                                const int dr_start = std::max(0, c - global_r0);
                                if (dr_start >= r_len) break;
                                const double* __restrict__ col_src =
                                    strip_f64 + static_cast<size_t>(c) * strip_h + local_r0;
                                for (int dr = dr_start; dr < r_len; ++dr) {
                                    tmp_d[dr][dc] = col_src[dr];
                                }
                            }
                            for (int dr = 0; dr < r_len; ++dr) {
                                const int pair1 = global_r0 + dr;
                                const int valid_c = std::min(c_len, pair1 + 1 - c0);
                                if (valid_c <= 0) continue;
                                const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                                const uint32_t* __restrict__ N_row = get_N_row(pair1) + c0;
                                const uint32_t* __restrict__ sm_c  = sub_miss_ptr + c0;
                                const size_t dst_off = row_off[static_cast<size_t>(r0 + dr)] + static_cast<size_t>(c0);
                                const double* __restrict__ td_row = tmp_d[dr];
                                if (out_f16) {
                                    gcta_grm_io::GrmFp16* __restrict__ g_dst = out_f16 + dst_off;
                                    const bool has_diag = (c0 + valid_c - 1 == pair1);
                                    const int off_end = has_diag ? (valid_c - 1) : valid_c;
                                    double off_sq = 0.0;
                                    for (int dc = 0; dc < off_end; ++dc) {
                                        const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                        const float v = sub_N
                                            ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight)
                                            : 0.0f;
                                        g_dst[dc] = gcta_grm_io::f32_to_fp16(v);
                                        const double vd = static_cast<double>(v);
                                        off_sq += vd * vd;
                                    }
                                    tr_K2_strip += 2.0 * off_sq;
                                    if (has_diag) {
                                        const uint32_t sub_N = N_row[off_end] + sub_miss1 - sm_c[off_end];
                                        const float dv = sub_N
                                            ? static_cast<float>((td_row[off_end] / static_cast<double>(sub_N)) * mtd_weight)
                                            : 0.0f;
                                        g_dst[off_end] = gcta_grm_io::f32_to_fp16(dv);
                                        fp16_diag_[static_cast<size_t>(pair1)] = dv;
                                        const double dd = static_cast<double>(dv);
                                        tr_K_strip += dd;
                                        tr_K2_strip += dd * dd;
                                    }
                                } else {
                                    float* __restrict__ g_dst = out_f32 + dst_off;
                                    for (int dc = 0; dc < valid_c; ++dc) {
                                        const uint32_t sub_N = N_row[dc] + sub_miss1 - sm_c[dc];
                                        g_dst[dc] = sub_N
                                            ? static_cast<float>((td_row[dc] / static_cast<double>(sub_N)) * mtd_weight)
                                            : 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }

                if (out_f16) {
                    fp16_trace_K_  += tr_K_strip;
                    fp16_trace_K2_ += tr_K2_strip;
                }

                if (!isSparse) {
                    grm_writer->submit_slot(strip_bytes);
                } else {
                    const int n_thr = (block_rows >= 64) ? omp_get_max_threads() : 1;
                    if (n_thr > 1) {
                        std::vector<std::string> thr_bufs(static_cast<size_t>(n_thr));
                        #pragma omp parallel num_threads(n_thr)
                        {
                            const int tid = omp_get_thread_num();
                            const int r_beg = (block_rows * tid) / n_thr;
                            const int r_end = (block_rows * (tid + 1)) / n_thr;
                            std::string& local_buf = thr_bufs[static_cast<size_t>(tid)];
                            for (int local_r = r_beg; local_r < r_end; ++local_r) {
                                const int pair1 = block_start + local_r;
                                const float* row = out_f32 + row_off[static_cast<size_t>(local_r)];
                                for (int i = 0; i <= pair1; ++i) {
                                    if (row[i] >= thresh) {
                                        std::format_to(std::back_inserter(local_buf),
                                            "{}\t{}\t{:.{}g}\n",
                                            pair1, i, row[i],
                                            std::numeric_limits<float>::digits10 + 2);
                                    }
                                }
                            }
                        }
                        for (const auto& tb : thr_bufs) {
                            if (!tb.empty() && fwrite(tb.data(), 1, tb.size(), sparse_out) != tb.size()) {
                                LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                            }
                        }
                    } else {
                        sparse_buf.clear();
                        for (int local_r = 0; local_r < block_rows; ++local_r) {
                            const int pair1 = block_start + local_r;
                            const float* row = out_f32 + row_off[static_cast<size_t>(local_r)];
                            for (int i = 0; i <= pair1; ++i) {
                                if (row[i] >= thresh) {
                                    std::format_to(std::back_inserter(sparse_buf),
                                        "{}\t{}\t{:.{}g}\n",
                                        pair1, i, row[i],
                                        std::numeric_limits<float>::digits10 + 2);
                                }
                            }
                        }
                        if (!sparse_buf.empty() && fwrite(sparse_buf.data(), 1, sparse_buf.size(), sparse_out) != sparse_buf.size()) {
                            LOGGER.e(0, "Failed to write sparse GRM output [" + grm_path + "].");
                        }
                    }
                }
            }
        }

        // Write .grm.N.bin sequentially AFTER draining grm_writer (strictly 1 active write stream!).
        if (!isSparse && N_fd >= 0) {
            grm_writer->drain();
            const uint64_t dre = static_cast<uint64_t>(tile_re);
            const uint64_t drs = static_cast<uint64_t>(tile_rs);
            const size_t tile_total_elems = static_cast<size_t>((dre * (dre + 1ULL) - drs * (drs + 1ULL)) / 2ULL);

            if (zero_missing) {
                const float n_val = static_cast<float>(numValidMarkers);
                const size_t buf_floats = std::min(tile_total_elems, kConstNFloats);
                if (const_N_buf.size() < buf_floats || (buf_floats > 0 && const_N_buf[0] != n_val)) {
                    const_N_buf.assign(buf_floats, n_val);
                }
                size_t rem_bytes = tile_total_elems * sizeof(float);
                const size_t cap_bytes = buf_floats * sizeof(float);
                while (rem_bytes > 0) {
                    const size_t c = std::min(rem_bytes, cap_bytes);
                    gcta_io::write_exact_throwing(N_fd, const_N_buf.data(), c, N_path);
                    rem_bytes -= c;
                }
            } else {
                // Missing-data tile: pack row-major triangular N[] into float32 strips via single AsyncSeqWriter.
                gcta_io::AsyncSeqWriter n_writer(N_fd, N_path, gcta_grm_detail::kGrmAsyncSlotBytes,
                                                 /*take_ownership=*/false);
                const uint32_t valid_m = numValidMarkers;
                const size_t max_n_elems = gcta_grm_detail::kGrmAsyncSlotBytes / sizeof(float);
                int r_cur = tile_rs;
                while (r_cur < tile_re) {
                    const size_t e0 = static_cast<size_t>(r_cur) * (r_cur + 1) / 2;
                    const int r_end = gcta_grm_io::row_bound_for_cumulative(e0 + max_n_elems, r_cur, tile_re);
                    const size_t e1 = static_cast<size_t>(r_end) * (r_end + 1) / 2;
                    const size_t chunk_bytes = (e1 - e0) * sizeof(float);

                    AlignedRawBuffer<char, 64>& slot = n_writer.acquire_slot();
                    if (slot.size() < chunk_bytes) slot.resize(chunk_bytes);
                    auto* dst_base = reinterpret_cast<float*>(slot.data());

                    #pragma omp parallel for schedule(dynamic, 64)
                    for (int pair1 = r_cur; pair1 < r_end; ++pair1) {
                        const uint32_t sub_miss1 = valid_m - sub_miss_ptr[pair1];
                        const uint32_t* __restrict__ N_row = get_N_row(pair1);
                        const size_t dst_off = static_cast<size_t>(pair1) * (pair1 + 1) / 2 - e0;
                        float* __restrict__ n_dst = dst_base + dst_off;
                        #pragma omp simd
                        for (int c = 0; c <= pair1; ++c) {
                            n_dst[c] = static_cast<float>(N_row[c] + sub_miss1 - sub_miss_ptr[c]);
                        }
                    }
                    n_writer.submit_slot(chunk_bytes);
                    r_cur = r_end;
                }
                n_writer.finish();
            }
        }
    } catch (const std::exception& e) {
        LOGGER.e(0, std::string("flush_grm_tile failed: ") + e.what());
    }
}

// Unified packed triangular N_thread (identical layout for tiled and dense paths).
void GRM::N_thread(int grm_index_from, int grm_index_to, const uintptr_t* cur_cmask){
    uint32_t* const N_base = std::assume_aligned<64>(N.get());
    const uint64_t rs_u = static_cast<uint64_t>(grm_tile_rs);
    const uint64_t startPos = (static_cast<uint64_t>(grm_index_from) + 1ULL + rs_u)
                            * static_cast<uint64_t>(grm_index_from - grm_tile_rs) / 2ULL;

    uint32_t *po_N = N_base + startPos;
    const uintptr_t *p_cmask1 = cur_cmask + grm_index_from;
    for (int index_pair1 = grm_index_from; index_pair1 != grm_index_to + 1; index_pair1++) {
        const uintptr_t *p_cmask2 = cur_cmask;
        uintptr_t cmask1 = *p_cmask1++;
        if (cmask1) {
            int count = index_pair1 + 1;
            int index_pair2 = 0;
            for (; index_pair2 + 3 < count; index_pair2 += 4) {
                uintptr_t c0 = p_cmask2[0];
                uintptr_t c1 = p_cmask2[1];
                uintptr_t c2 = p_cmask2[2];
                uintptr_t c3 = p_cmask2[3];
                po_N[0] += std::popcount(cmask1 & c0);
                po_N[1] += std::popcount(cmask1 & c1);
                po_N[2] += std::popcount(cmask1 & c2);
                po_N[3] += std::popcount(cmask1 & c3);
                p_cmask2 += 4;
                po_N += 4;
            }
            for (; index_pair2 < count; index_pair2++) {
                uintptr_t cmask = cmask1 & (*p_cmask2++);
                if (cmask) *po_N += std::popcount(cmask);
                po_N++;
            }
        } else {
            po_N += index_pair1 + 1;
        }
    }
}

vector<uint32_t> GRM::divide_parts_mem(uint32_t n_sample, uint32_t num_parts){
    int parts = num_parts;
    vector<double> n_part(parts);
    vector<double> n_each_part(parts);
    n_part[0] = 1;
    n_each_part[0] = 1;
    for (int i = 1; i < parts; i++) {
        double previous_n = n_part[i - 1];
        double cur_n = (sqrt(36.0 * previous_n * previous_n + 100.0) - 6.0 * previous_n) / 10.0;
        n_each_part[i] = cur_n;
        n_part[i] = previous_n + cur_n;
    }

    double x1 = n_sample / n_part[parts - 1];
    vector<uint32_t> div_parts(parts);
    uint32_t total_num = 0;
    for (int i = 0; i < parts - 1; i++) {
        uint32_t cur_n = round(n_each_part[i] * x1);
        total_num = cur_n + total_num;
        div_parts[i] = total_num - 1;
    }
    div_parts[parts - 1] = n_sample - 1;
    return div_parts;
}

vector<uint32_t> GRM::divide_parts(uint32_t from, uint32_t to, uint32_t num_parts){
    vector<uint64_t> num_indi_grms;
    num_indi_grms.reserve(to - from + 1);
    for (uint64_t index = from; index != to + 1; index++) {
        num_indi_grms.push_back((index + 2 + from) * (index + 1 - from) / 2);
    }

    uint64_t indi_parts = num_indi_grms[to - from] / num_parts;
    vector<uint32_t> parts;
    parts.reserve(num_parts);
    vector<uint64_t>::iterator upper;
    for (uint32_t index = 1; index <= num_parts; index++) {
        upper = std::upper_bound(num_indi_grms.begin(), num_indi_grms.end(), indi_parts * index);
        if (upper != num_indi_grms.begin()) {
            upper--;
            if (*upper < indi_parts * index) {
                upper++;
            }
        }
        parts.push_back(upper - num_indi_grms.begin() + from);
    }
    parts.erase(unique(parts.begin(), parts.end()), parts.end());
    return parts;
}

void GRM::processMakeGRM(bool is_homogametic){
    is_homogametic_ = is_homogametic;
    use_int8_grm = bBLAS && isMtd && !isDominance && !is_homogametic
                   && (geno->getGenoFormat() == "BED")
                   && (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);
    if (!n_markers_explicit_ && !use_int8_grm && nMarkerBlock > 1024) {
        nMarkerBlock = 1024;
    }

    const vector<uint32_t> processIndex = is_homogametic
        ? marker->get_extract_index_X()
        : marker->get_extract_index_autosome();

    gbufitems.assign(static_cast<size_t>(nMarkerBlock), GenoBufItem{});
    validIndexBuf.reserve(static_cast<size_t>(nMarkerBlock));
    sampleMissBuf.reset();

    constexpr int kTile = kGrmStripRows;
    const int max_tile_dim = std::min(kTile, std::max(1, grm_n));
    const size_t geno_block_elems = static_cast<size_t>(nMarkerBlock) * static_cast<size_t>(stdGenoLD);

    if (use_int8_grm) {
        grm_bytes_std_geno = sizeof(int8_t) * static_cast<size_t>(grm_n);
        stdGeno_i8.reset();
        stdGeno_i8_cols_ = 0;
        const int eff_block_cap = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(nMarkerBlock), std::max<size_t>(1, processIndex.size())));
        const int max_ldK = std::max(64, (eff_block_cap + 63) / 64 * 64);
        const size_t kmajor_elems = static_cast<size_t>(max_ldK) * static_cast<size_t>(grm_n);
        this->num_byte_geno = sizeof(int8_t) * static_cast<uint64_t>(kmajor_elems);
        stdGeno_i8_kmajor_ = make_aligned_uninitialized<int8_t, 64>(kmajor_elems);
        if (!stdGeno_i8_kmajor_) {
            LOGGER.e(0, "can't allocate enough memory for the transposed int8 genotype buffer.");
        }
        grm_i32_tile_.reset();
        mu_vec_.resize(static_cast<size_t>(nMarkerBlock));
        r_raw_.resize(static_cast<size_t>(grm_n));
        r_tilde_.resize(static_cast<size_t>(grm_n));
        marker_miss_offset_.resize(static_cast<size_t>(nMarkerBlock) + 1);
        sample_miss_offset_.resize(static_cast<size_t>(grm_n) + 1);
    } else {
        grm_bytes_std_geno = sizeof(float) * static_cast<size_t>(grm_n);
        this->num_byte_geno = sizeof(float) * static_cast<uint64_t>(geno_block_elems);
        stdGeno = make_aligned_uninitialized<float, 64>(geno_block_elems);
        if (!stdGeno) {
            LOGGER.e(0, "can't allocate enough memory for the genotype buffer.");
        }
        grm_f32_tile_buf_.resize(static_cast<size_t>(max_tile_dim) * static_cast<size_t>(max_tile_dim));
    }

    vector<function<void (uintptr_t *, std::span<const uint32_t>)>> callBacks;
    callBacks.push_back([this](uintptr_t *buf, std::span<const uint32_t> idx){
        calculate_GRM_blas(buf, idx);
    });
    geno->setGRMMode(true, isDominance);
    const bool isSTD = !isMtd;
    sd.clear();
    sd.reserve(processIndex.size());
    sub_miss.assign(index_keep.size() + 64, 0);
    numValidMarkers = 0;
    finished_marker = 0;
    grm_any_missing_seen_ = false;
    grm_skip_global_state = false;
    tile1_valid_extract_.clear();
    tile1_valid_mean_.clear();
    tile1_valid_has_miss_.clear();
    tile1_cursor_ = 0;
    use_tile1_cache_ = false;

    const bool allow_i32_mode = use_int8_grm && (processIndex.size() <= kMaxInt32Markers);
    const double initial_bpe = allow_i32_mode ? 4.0 : 8.0;
    const int full_rs = static_cast<int>(part_keep_indices.first);
    const int full_re = static_cast<int>(part_keep_indices.second) + 1;

    grm_tiling_enabled = grm_tile_budget_bytes > 0.0;
    if (grm_tiling_enabled) {
        const double budget_elems = grm_tile_budget_bytes / initial_bpe;
        if (gcta_grm_detail::grm_tile_budget_covers_whole(full_rs, full_re, budget_elems)) {
            LOGGER.info(0, "--GRM-tile-budget ({:.3f} GB) covers the whole GRM in one tile ({:.0f} B/elem); using the dense path instead.",
                        grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0), initial_bpe);
            grm_tiling_enabled = false;
        }
    }

    if (bBLAS && grm_tiling_enabled) {
        float thresh = -99.0f;
        bool isSparse = false;
        if (sparse_cutoff.has_value()) {
            thresh   = *sparse_cutoff;
            isSparse = true;
            LOGGER.i(0, "Computing sparse GRM with a cutoff " + to_string(thresh) + "...");
        } else {
            LOGGER.i(0, write_fp16_ ? "Computing tiled native FP16 GRM..." : "Computing tiled GRM...");
        }

        const std::string grm_path = isSparse ? (o_name + ".grm.sp") : (o_name + (write_fp16_ ? ".grm.f16" : ".grm.bin"));
        const std::string N_path   = o_name + ".grm.N.bin";
        const uint64_t n_out = static_cast<uint64_t>(num_individual);
        const uint64_t body_off = write_fp16_ ? gcta_grm_io::grm_f16_body_offset(n_out) : 0ULL;

        UniqueFile sparse_out;
        std::unique_ptr<gcta_io::AsyncSeqWriter> grm_writer;
        gcta_grm_detail::GrmUniqueFd N_fd;
        if (isSparse) {
            sparse_out = open_file(grm_path, "wb");
            if (!sparse_out) LOGGER.error(0, "can't open {} to write", grm_path);
            setvbuf(sparse_out.get(), nullptr, _IOFBF, 4 << 20);
        } else {
            try {
                grm_writer = std::make_unique<gcta_io::AsyncSeqWriter>(
                    grm_path, gcta_grm_detail::kGrmAsyncSlotBytes, static_cast<off_t>(body_off));
                if (write_fp16_) {
                    gcta_grm_io::write_grm_f16_draft_prefix(*grm_writer, n_out);
                    fp16_diag_.assign(static_cast<size_t>(n_out), 0.0f);
                    fp16_trace_K_ = 0.0;
                    fp16_trace_K2_ = 0.0;
                }
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("processMakeGRM tiled open failed: ") + e.what());
            }
            // For classic .grm.bin, open .grm.N.bin upfront; for .grm.f16, open lazily
            // after Tile 0 ONLY if missing genotypes were observed across the cohort!
            if (!write_fp16_) {
                N_fd = gcta_grm_detail::open_posix_write_trunc(N_path);
                if (!N_fd) LOGGER.error(0, "can't open {} to write", N_path);
            }
        }

        const int num_thread = omp_get_max_threads();

        {
            const double budget_elems = grm_tile_budget_bytes / initial_bpe;
            vector<std::pair<int, int>> est_ranges;
            int sim_rs = full_rs;
            while (sim_rs < full_re) {
                const int sim_re = gcta_grm_detail::solve_tile_budget_end(sim_rs, full_re, budget_elems);
                if (sim_re <= sim_rs) {
                    const uint64_t need_bytes = static_cast<uint64_t>(sim_rs + 1) * static_cast<uint64_t>(initial_bpe);
                    LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                                 sim_rs,
                                 need_bytes / (1024.0 * 1024.0 * 1024.0),
                                 initial_bpe,
                                 grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
                }
                est_ranges.push_back({sim_rs, sim_re});
                sim_rs = sim_re;
            }
            LOGGER.info(0, "Block-tiled GRM (strip-packed, budget {:.3f} GB, initial {:.0f} B/elem): {} tile(s), rows/tile ranging {}-{}.",
                        grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0),
                        initial_bpe,
                        est_ranges.size(),
                        est_ranges.front().second - est_ranges.front().first,
                        est_ranges.back().second - est_ranges.back().first);
        }

        std::vector<float> flush_const_N;
        std::string flush_sparse_buf;

        double mtd_weight = 1.0;
        bool logged_missing_rebudget = false;
        int rs = full_rs;

        while (rs < full_re) {
            const double cur_bpe = grm_any_missing_seen_ ? 12.0 : initial_bpe;
            if (grm_any_missing_seen_ && !logged_missing_rebudget && rs > full_rs) {
                logged_missing_rebudget = true;
                LOGGER.info(0, "  Missing genotypes detected; adapting remaining tile budget to 12 B/elem (double GRM + N).");
            }
            const int re = gcta_grm_detail::solve_tile_budget_end(rs, full_re, grm_tile_budget_bytes / cur_bpe);
            if (re <= rs) {
                const uint64_t need_bytes = static_cast<uint64_t>(rs + 1) * static_cast<uint64_t>(cur_bpe);
                LOGGER.error(0, "--GRM-tile-budget is too small: the next row (sample offset {}) alone needs {:.3f} GB ({:.0f} B/elem), but the budget is only {:.3f} GB. Increase --GRM-tile-budget.",
                             rs,
                             need_bytes / (1024.0 * 1024.0 * 1024.0),
                             cur_bpe,
                             grm_tile_budget_bytes / (1024.0 * 1024.0 * 1024.0));
            }

            grm_tile_rs   = rs;
            grm_tile_re   = re;
            grm_tile_rows = re - rs;
            grm_tile_cols = re;

            bool known_tile_missing = false;
            if (grm_skip_global_state && grm_any_missing_seen_) {
                for (int s = 0; s < re; ++s) {
                    if (sub_miss[static_cast<size_t>(s)] > 0) {
                        known_tile_missing = true;
                        break;
                    }
                }
            }
            const bool start_i32 = allow_i32_mode && !known_tile_missing;

            reset_grm_strips();
            N.reset();

            if (start_i32) {
                allocate_grm_strips(/*as_i32=*/true);
                grm_is_i32_ = true;
                grm_tile_i32_markers_ = 0;
                R_total_.assign(static_cast<size_t>(grm_tile_cols), 0.0);
                S_total_ = 0.0;
            } else {
                allocate_grm_strips(/*as_i32=*/false);
                grm_is_i32_ = false;
            }

            const double drs = rs, dre = re;
            const double tile_elems = (dre * (dre + 1.0) - drs * (drs + 1.0)) / 2.0;
            const double est_bpe = start_i32 ? 4.0 : (known_tile_missing ? 12.0 : 8.0);
            const double tile_gb = tile_elems * est_bpe / (1024.0 * 1024.0 * 1024.0);
            LOGGER.info(0, "  Tile rows {}-{} ({:.2f} GB strip-packed {})", rs, re - 1, tile_gb,
                        start_i32 ? "int32 GRM" : (known_tile_missing ? "double GRM+N" : "double GRM"));

            grm_first_accum = true;

            index_grm_pairs.clear();
            vector<uint32_t> tile_parts = divide_parts(static_cast<uint32_t>(rs), static_cast<uint32_t>(re - 1), static_cast<uint32_t>(num_thread));
            index_grm_pairs.push_back(std::make_pair(rs, static_cast<int>(tile_parts[0])));
            for (size_t i = 1; i < tile_parts.size(); ++i)
                index_grm_pairs.push_back(std::make_pair(static_cast<int>(tile_parts[i-1] + 1), static_cast<int>(tile_parts[i])));

            const bool is_first_tile = !grm_skip_global_state;
            if (is_first_tile) {
                use_tile1_cache_ = false;
                if (use_int8_grm) {
                    tile1_valid_extract_.reserve(processIndex.size());
                    tile1_valid_mean_.reserve(processIndex.size());
                    tile1_valid_has_miss_.reserve(processIndex.size());
                }
                geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
            } else if (numValidMarkers > 0) {
                if (use_int8_grm && !tile1_valid_extract_.empty()) {
                    // Tiles 2..T (INT8 BED): stream ONLY valid markers recorded on Tile 1,
                    // skipping invalid/filtered markers and the redundant stats_only pass!
                    tile1_cursor_ = 0;
                    use_tile1_cache_ = true;
                    geno->loopDouble(tile1_valid_extract_, nMarkerBlock, true, true, isSTD, true, callBacks);
                    use_tile1_cache_ = false;
                } else {
                    // FP32 path (!use_int8_grm): preserve exact batch boundaries so cblas_sgemm
                    // float32->double widening remains bit-identical to dense GRM.
                    use_tile1_cache_ = false;
                    geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
                }
            }

            grm_skip_global_state = true;

            if (is_first_tile) {
                if (isMtd) {
                    double weight = 0.0;
                    if (!isDominance)
                        for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i];
                    else
                        for (uint32_t i = 0; i < numValidMarkers; i++) weight += sd[i] * sd[i];
                    if (weight > 0.0) mtd_weight = static_cast<double>(numValidMarkers) / weight;
                }
                if (!isSparse && write_fp16_ && grm_any_missing_seen_) {
                    N_fd = gcta_grm_detail::open_posix_write_trunc(N_path);
                    if (!N_fd) LOGGER.error(0, "can't open {} to write", N_path);
                }
            }

            flush_grm_tile(grm_writer.get(), N_fd.get(), sparse_out.get(), grm_path, N_path,
                           thresh, isSparse, mtd_weight, flush_const_N, flush_sparse_buf);

            reset_grm_strips();
            N.reset();
            rs = re;
        }

        if (!isSparse) {
            try {
                if (write_fp16_) {
                    gcta_grm_io::GrmFp16Header final_hdr{};
                    std::memcpy(final_hdr.magic, gcta_grm_io::kGrmF16MagicFinal, sizeof(final_hdr.magic));
                    final_hdr.version = gcta_grm_io::kGrmF16Version;
                    final_hdr.dtype = gcta_grm_io::kGrmF16DtypeFp16;
                    final_hdr.header_bytes = gcta_grm_io::kGrmF16HeaderSize;
                    final_hdr.flags = gcta_grm_io::kGrmF16FlagFinalized |
                                      gcta_grm_io::kGrmF16FlagHasTraces |
                                      gcta_grm_io::kGrmF16FlagHasDiagF32 |
                                      (!grm_any_missing_seen_ ? gcta_grm_io::kGrmF16FlagZeroMissing : 0u) |
                                      (isMtd ? gcta_grm_io::kGrmF16FlagVanRaden : 0u) |
                                      (isDominance ? gcta_grm_io::kGrmF16FlagDominance : 0u) |
                                      (is_homogametic_ ? gcta_grm_io::kGrmF16FlagHomogametic : 0u);
                    final_hdr.n = n_out;
                    final_hdr.valid_markers = numValidMarkers;
                    if (!grm_any_missing_seen_) {
                        final_hdr.m_snps_mean = static_cast<double>(numValidMarkers);
                    } else {
                        double miss_sum = 0.0;
                        for (uint64_t i = 0; i < n_out; ++i) miss_sum += static_cast<double>(sub_miss[static_cast<size_t>(i)]);
                        final_hdr.m_snps_mean = (n_out > 0)
                            ? (static_cast<double>(numValidMarkers) - miss_sum / static_cast<double>(n_out))
                            : 0.0;
                    }
                    final_hdr.id_hash = grm_id_hash_;
                    final_hdr.trace_K = fp16_trace_K_;
                    final_hdr.trace_K2 = fp16_trace_K2_;
                    final_hdr.diag_offset = gcta_grm_io::kGrmF16HeaderSize;
                    final_hdr.body_offset = body_off;
                    final_hdr.part_row_start = 0;
                    final_hdr.part_row_end = n_out;
                    final_hdr.mtd_weight = mtd_weight;
                    gcta_grm_io::finalize_grm_f16_file(*grm_writer, final_hdr, fp16_diag_.data(), n_out);
                } else {
                    grm_writer->finish();
                }
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("Failed finalizing [") + grm_path + "]: " + e.what());
            }
        }
        grm_writer.reset();
        N_fd.reset();
        sparse_out.reset();

        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        if (isSparse) {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.sp]");
        } else if (write_fp16_) {
            LOGGER.i(0, "Native FP16 GRM has been saved in the file [" + grm_path + "]" +
                        (!grm_any_missing_seen_
                             ? " (zero missing genotypes; .grm.N.bin omitted, m_snps=" + to_string(numValidMarkers) + " in header)"
                             : " (missingness present; per-pair counts in [" + N_path + "])"));
        } else {
            LOGGER.i(0, "GRM has been saved in the file [" + o_name + ".grm.bin]");
            LOGGER.i(0, "Number of SNPs in each pair of individuals has been saved in the file [" + o_name + ".grm.N.bin]");
        }
    } else {
        // Non-tiled dense path (!grm_tiling_enabled): single pass with 4096-row strip-packed storage.
        grm_tile_rs   = full_rs;
        grm_tile_re   = full_re;
        grm_tile_rows = grm_m;
        grm_tile_cols = grm_n;

        reset_grm_strips();
        N.reset();

        if (allow_i32_mode) {
            allocate_grm_strips(/*as_i32=*/true);
            grm_is_i32_ = true;
            grm_tile_i32_markers_ = 0;
            R_total_.assign(static_cast<size_t>(grm_n), 0.0);
            S_total_ = 0.0;
        } else {
            allocate_grm_strips(/*as_i32=*/false);
            grm_is_i32_ = false;
        }
        grm_first_accum = true;

        LOGGER << "Computing dense GRM..." << std::endl;
        geno->loopDouble(processIndex, nMarkerBlock, true, true, isSTD, true, callBacks);
        LOGGER << "  Used " << numValidMarkers << " valid SNPs." << std::endl;
        deduce_GRM();
        reset_grm_strips();
        N.reset();
    }

    gbufitems.clear();
    tile1_valid_extract_.clear();
    tile1_valid_mean_.clear();
    tile1_valid_has_miss_.clear();
    stdGeno.reset();
    stdGeno_i8.reset();
    stdGeno_i8_cols_ = 0;
    stdGeno_i8_kmajor_.reset();
    sampleMissBuf.reset();
    grm_f32_tile_buf_.reset();
    grm_i32_tile_.reset();
    grm_i32_diag_scratch_.reset();
    i8_reordered_B_.reset();
    geno->setGRMMode(false, false);
}

void GRM::processMakeGRMX(){
    processMakeGRM(true);
}

void GRM::runMake(const gcta_cli::GrmMakeOptions& opts, const std::string& out_prefix) {
    if (opts.homogametic) {
        LOGGER.i(0, "Note: this function treats homogametic chromosomes as non-PAR regions.");
    } else {
        LOGGER.i(0, "Note: GRM is computed using the SNPs on the autosomes.");
    }
    const bool has_explicit_chr = opts.marker_filter.autosome || opts.marker_filter.autosome_sex ||
                                  opts.marker_filter.chr_homogametic || !opts.marker_filter.chr.empty();
    Pheno pheno(opts.geno, opts.sample_filter);
    Marker marker(opts.geno, opts.sample_filter, opts.marker_filter,
                  /*default_autosome=*/!opts.homogametic && !has_explicit_chr,
                  /*force_chr_homogametic=*/opts.homogametic && !has_explicit_chr);

    const bool is_bed_input = !opts.geno.bfile.empty() || !opts.geno.mbfile.empty();
    const bool is_pgen_input = !opts.geno.pfile.empty() || !opts.geno.bpfile.empty() ||
                               !opts.geno.mpfile.empty() || !opts.geno.mbpfile.empty();
    const bool eligible_clean_int8 =
        !opts.legacy_engine &&
        opts.alg == gcta_cli::GrmAlg::VanRaden &&
        !opts.dominance &&
        !opts.homogametic &&
        (is_bed_input || is_pgen_input) &&
        opts.geno.update_freq.empty() &&
        (std::getenv("GCTA_DISABLE_INT8_GRM") == nullptr);

    if (eligible_clean_int8) {
        if (is_bed_input) {
            gcta_grm::run_bed_int8_grm(pheno, marker, opts, out_prefix);
            return;
        }

        std::vector<std::string> pgen_files;
        if (!opts.geno.pfile.empty()) {
            const std::string f = opts.geno.pfile + ".pgen";
            requireFileReadable("--pfile", f);
            pgen_files.push_back(f);
        } else if (!opts.geno.bpfile.empty()) {
            const std::string f = opts.geno.bpfile + ".pgen";
            requireFileReadable("--bpfile", f);
            pgen_files.push_back(f);
        } else if (!opts.geno.mpfile.empty()) {
            pgen_files = expandFileList(opts.geno.mpfile, ".pgen");
        } else if (!opts.geno.mbpfile.empty()) {
            pgen_files = expandFileList(opts.geno.mbpfile, ".pgen");
        }

        const uint32_t expected_samples = pheno.count_raw();
        std::vector<pgen_hardcall::PgenFileIndex> pgen_indices;
        pgen_indices.reserve(pgen_files.size());
        bool all_native_hardcall = !pgen_files.empty();
        std::string fallback_file;
        std::string fallback_reason;

        for (size_t i = 0; i < pgen_files.size(); ++i) {
            const MarkerParam mp = marker.getMarkerParams(static_cast<int>(i));
            pgen_hardcall::PgenFileIndex idx;
            try {
                idx = pgen_hardcall::load_pgen_index(pgen_files[i], expected_samples, mp.rawCountSNP);
            } catch (const std::exception& e) {
                LOGGER.e(0, std::string("Failed to read PGEN header [") + pgen_files[i] + "]: " + e.what());
            }
            if (idx.native_hardcall_ok &&
                (idx.raw_sample_ct != expected_samples || idx.raw_variant_ct != mp.rawCountSNP)) {
                LOGGER.e(0, "PGEN header dimension mismatch for [" + pgen_files[i] + "] (expected " +
                                std::to_string(expected_samples) + " samples x " +
                                std::to_string(mp.rawCountSNP) + " variants, found " +
                                std::to_string(idx.raw_sample_ct) + " samples x " +
                                std::to_string(idx.raw_variant_ct) + " variants).");
            }
            if (!idx.native_hardcall_ok && all_native_hardcall) {
                all_native_hardcall = false;
                fallback_file = pgen_files[i];
                fallback_reason = idx.fallback_reason;
            }
            pgen_indices.push_back(std::move(idx));
        }

        if (all_native_hardcall) {
            gcta_grm::run_bed_int8_grm(pheno, marker, opts, out_prefix, std::move(pgen_indices));
            return;
        }
        if (opts.impute_mean) {
            LOGGER.e(0, "--impute-mean requires native hardcall BED/PGEN input; [" + fallback_file +
                            "] cannot use the clean-room INT8 GRM engine (" +
                            (fallback_reason.empty() ? "dosages or multiallelic variants present" : fallback_reason) + ").");
        }
    }
    if (opts.impute_mean) {
        LOGGER.e(0, "--impute-mean requires the clean-room BED/PGEN INT8 GRM engine (unset GCTA_DISABLE_INT8_GRM).");
    }

    GRM grm(&pheno, &marker, opts, out_prefix);
    grm.processMakeGRM(opts.homogametic);
}

void GRM::runMerge(const gcta_cli::GrmMergeOptions& opts, const std::string& out_prefix) {
    GRM grm("", "", "", out_prefix, false, opts.grm_fp16);
    grm.merge_grm(opts.mgrm, out_prefix, opts.budget_gb, opts.grm_fp16);
}

void GRM::runPrune(const gcta_cli::GrmPruneOptions& opts, const std::string& out_prefix) {
    const bool no_grm = opts.singleton.has_value();
    const float cutoff = static_cast<float>(no_grm ? *opts.singleton : *opts.cutoff);
    const bool detail = no_grm || opts.cutoff_detail;
    GRM grm(opts.grm, opts.keep, opts.remove, out_prefix, detail, opts.grm_fp16);
    grm.cut_rel(cutoff, no_grm);
}

void GRM::runSparse(const gcta_cli::GrmSparseOptions& opts, const std::string& out_prefix) {
    GRM grm(opts.grm, opts.keep, opts.remove, out_prefix, false, opts.grm_fp16);
    float set_val = 0.0f;
    float* p = nullptr;
    if (opts.set_value.has_value()) {
        set_val = static_cast<float>(*opts.set_value);
        p = &set_val;
    }
    grm.prune_fam(static_cast<float>(opts.cutoff), !opts.dense, p);
}

void GRM::runSubtract(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix) {
    GRM grm("", opts.keep, opts.remove, out_prefix, false, opts.grm_fp16);
    grm.subtract_grm(opts.mgrm, out_prefix, opts.grm_fp16);
}

void GRM::runUnify(const gcta_cli::GrmCombineOptions& opts, const std::string& out_prefix) {
    GRM grm("", opts.keep, opts.remove, out_prefix, false, opts.grm_fp16);
    grm.unify_grm(opts.mgrm, out_prefix, opts.grm_fp16);
}
