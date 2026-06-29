#ifndef AUTOMATIC_ANNOTATION_STATIC_TRACK_ID_UTILS_HPP
#define AUTOMATIC_ANNOTATION_STATIC_TRACK_ID_UTILS_HPP

#include <cstddef>
#include <string>

namespace automatic_annotation {

std::string FormatStaticTrackName(std::size_t instance_index);

std::string FormatStaticTrackId(const std::string& object_type,
                                std::size_t instance_index);

}  // namespace automatic_annotation

#endif  // AUTOMATIC_ANNOTATION_STATIC_TRACK_ID_UTILS_HPP
