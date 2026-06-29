#include "automatic_annotation/static_track_id_utils.hpp"

#include <cctype>
#include <iomanip>
#include <sstream>

namespace automatic_annotation {
namespace {

std::string NormalizeClassNameForTrackId(std::string object_type) {
  for (char& ch : object_type) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  return object_type;
}

}  // namespace

std::string FormatStaticTrackName(std::size_t instance_index) {
  return std::to_string(instance_index);
}

std::string FormatStaticTrackId(const std::string& object_type,
                                std::size_t instance_index) {
  std::ostringstream oss;
  oss << "static_" << NormalizeClassNameForTrackId(object_type) << "_"
      << std::setw(6) << std::setfill('0') << instance_index;
  return oss.str();
}

}  // namespace automatic_annotation
