#include "OptionIO.h"
#include "Logger.h"
#include "utils.hpp"
#include <iostream>
#include <sstream>
#include <set>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

using std::to_string;

void addOneFileOption(string key_store, string append_string, string key_name, map<string, vector<string>> options_in, map<string,string>& options) {
    if(options_in.find(key_name) != options_in.end()){
        if(options_in[key_name].size() == 1){
            options[key_store] = options_in[key_name][0] + append_string;
        }else if(options_in[key_name].size() > 1){
            options[key_store] = options_in[key_name][0] + append_string;
            LOGGER.w(0, "There are multiple " + key_name + ". Only the first item will be used in the analysis." );
        }else{
            LOGGER.e(0, "no " + key_name + " parameter found");
        }
        if(!checkFileReadable(options[key_store])){
            LOGGER.e(0, key_name + " " + options[key_store] + " not found");
        }
    }
}

void requireFileReadable(const string& flag_name, const string& path) {
    if (!checkFileReadable(path)) {
        LOGGER.e(0, flag_name + " " + path + " not found");
    }
}

vector<string> expandFileList(const string& list_filename, const string& suffix) {
    requireFileReadable("file list", list_filename);

    vector<string> m_files;
    std::ifstream m_file(list_filename);
    string line;
    while (std::getline(m_file, line)) {
        std::istringstream iss(line);
        string temp_fname;
        if (!(iss >> temp_fname)) {
            continue;
        }
        boost::replace_all(temp_fname, "\r", "");
        m_files.push_back(temp_fname);
    }

    if (m_files.empty()) {
        LOGGER.e(0, "no item in [" + list_filename + "].");
    }

    std::set<string> unique_mfiles;
    auto ends = std::remove_if(m_files.begin(), m_files.end(),
                               [&unique_mfiles](const string& item) {
                                   if (unique_mfiles.find(item) != unique_mfiles.end()) {
                                       return true;
                                   }
                                   unique_mfiles.insert(item);
                                   return false;
                               });
    m_files.erase(ends, m_files.end());

    vector<string> out_files;
    out_files.reserve(m_files.size());
    for (const auto& temp_fname : m_files) {
        if (temp_fname.size() >= suffix.size() &&
            temp_fname.compare(temp_fname.size() - suffix.size(), suffix.size(), suffix) == 0) {
            out_files.push_back(temp_fname);
        } else {
            out_files.push_back(temp_fname + suffix);
        }
    }
    return out_files;
}

void addMFileListsOption(string key_store, string append_string, string key_name, map<string, vector<string>> options_in, map<string,string>& options){
    string list_filename;
    if(options_in.find(key_name) != options_in.end()){
        if(options_in[key_name].size() == 1){
            list_filename = options_in[key_name][0];
        }else if(options_in[key_name].size() > 1){
            list_filename = options_in[key_name][0];
            LOGGER.w(0, "There are multiple " + key_name + ". Only the first item will be used in the analysis." );
        }else{
            LOGGER.e(0, "no " + key_name + " parameter found");
        }

        const vector<string> mc_files = expandFileList(list_filename, append_string);
        options[key_store] = boost::algorithm::join(mc_files, "\t");
    }
}

// get the header line
// return true:   success, head empty, means no head line; ifstream will be set to the first line of contents 
//        false:  fail to get header. ncol = 0: all comment header, or contains blank line in the header;
//                                    ncol != 0: head is inconsistent with contents
// 
bool getListHeaderNew(std::ifstream *input, bool &ioerr, int &ncol, vector<string> &head, int &nHeader){
    uint64_t startByte = 0;
    ioerr = false;
    ncol = 0;
    string line;
    uint64_t prePos = 0;
    string preLine;
    if(!input->good()){
        ioerr = true;
        return false;
    }
    std::size_t found = std::string::npos;
    int curReadLine = 0;
    bool got_line = false;
    while(std::getline(*input, line)){
        got_line = true;
        curReadLine++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        found = line.find_first_not_of("\r\t ");
        if(found != std::string::npos && line.at(found) == '#'){
            preLine = line;
            prePos = static_cast<uint64_t>(input->tellg());
            continue;
        }else{
            break;
        }
    }

    if(!got_line || found == std::string::npos || line.at(found) == '#'){
        nHeader = curReadLine;
        return false;
    }

    vector<std::string_view> line_elements;
    gcta_text_io::split_ws_exact(line, line_elements);
    int ncol_con = static_cast<int>(line_elements.size());

    ncol = ncol_con;
    startByte = prePos;
    nHeader = curReadLine - 1;

    if(!preLine.empty()){
        vector<std::string_view> head_views;
        gcta_text_io::split_ws_exact(preLine, head_views);
        int ncol_head = static_cast<int>(head_views.size());
        if(ncol_head != ncol_con){
            return false;
        }else{
            head.clear();
            head.reserve(head_views.size());
            for (const auto sv : head_views) head.emplace_back(sv);
        }
    }
    input->clear();
    input->seekg(static_cast<std::streamoff>(startByte));
    return true;
}

bool readTxtList(string fileName, int minFields, vector<string> &head, int &nHeader, map<int, vector<string>> &lists){
    gcta_text_io::FastLineReader reader;
    if (!reader.open(fileName)) {
        LOGGER.e(0, "can't read [" + fileName + "].");
        return false;
    }

    std::string_view line;
    std::string preLine;
    int curReadLine = 0;
    bool found_data = false;
    while (reader.next_line(line)) {
        curReadLine++;
        const size_t first_non_ws = line.find_first_not_of("\r\t ");
        if (first_non_ws != std::string_view::npos && line[first_non_ws] == '#') {
            preLine.assign(line);
            continue;
        }
        if (first_non_ws == std::string_view::npos) {
            nHeader = curReadLine;
            LOGGER.e(0, "blank header line or no valid text data.");
            return false;
        }
        found_data = true;
        break;
    }

    if (!found_data) {
        nHeader = curReadLine;
        LOGGER.e(0, "blank header line or no valid text data.");
        return false;
    }

    std::vector<std::string_view> toks;
    toks.reserve(16);
    gcta_text_io::split_ws_exact(line, toks);
    const int ncol = static_cast<int>(toks.size());
    nHeader = curReadLine - 1;

    head.clear();
    if (!preLine.empty()) {
        std::vector<std::string_view> head_toks;
        gcta_text_io::split_ws_exact(preLine, head_toks);
        if (static_cast<int>(head_toks.size()) != ncol) {
            LOGGER.e(0, "data are inconsistent with the headers.");
            return false;
        }
        head.reserve(head_toks.size());
        for (const auto sv : head_toks) head.emplace_back(sv);
    }

    if (ncol < minFields) {
        LOGGER.e(0, "the number of fields in [" + fileName + "] is less than " + to_string(minFields) + ".");
    }

    const size_t est_rows = (reader.file_size() > 0)
        ? static_cast<size_t>(reader.file_size() / std::max<size_t>(line.size() + 1, 8)) + 16
        : 0;
    lists.clear();
    std::vector<std::vector<std::string>*> col_ptrs(static_cast<size_t>(ncol));
    for (int i = 0; i < ncol; ++i) {
        auto& col = lists[i];
        if (est_rows > 0) col.reserve(est_rows);
        col_ptrs[static_cast<size_t>(i)] = &col;
    }

    int line_number = nHeader;
    auto push_row = [&]() {
        line_number++;
        if (static_cast<int>(toks.size()) < ncol) {
            LOGGER.e(0, "the file [" + fileName + "] contains different number of elements in line " + to_string(line_number) + ".");
        } else if (static_cast<int>(toks.size()) > ncol) {
            LOGGER.w(0, "the file [" + fileName + "] contains extra number of elements in line " + to_string(line_number) + ".");
        }
        for (int i = 0; i < ncol; ++i) {
            col_ptrs[static_cast<size_t>(i)]->emplace_back(toks[static_cast<size_t>(i)]);
        }
    };

    push_row();
    while (reader.next_line(line)) {
        gcta_text_io::split_ws_exact(line, toks);
        push_row();
    }
    return true;
}

bool checkFileReadable(string filename){
    std::ifstream f(filename.c_str());
    return f.good();
}

uint64_t getFileSize(FILE * file){
   if(file == NULL){
       return 0;
   }
   fseek(file, 0, SEEK_END);
   uint64_t f_size = ftell(file);
   rewind(file);
   return f_size;
}




