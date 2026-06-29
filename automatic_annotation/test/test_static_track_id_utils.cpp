#include <gtest/gtest.h>

#include "automatic_annotation/static_track_id_utils.hpp"

namespace automatic_annotation {
namespace {

TEST(StaticTrackIdUtilsTest, FormatsTrackNameAsGlobalInstanceIndex) {
  EXPECT_EQ(FormatStaticTrackName(1), "1");
  EXPECT_EQ(FormatStaticTrackName(42), "42");
}

TEST(StaticTrackIdUtilsTest, FormatsStableTrackIdWithNormalizedClassName) {
  EXPECT_EQ(FormatStaticTrackId("Distance Marker", 1),
            "static_Distance_Marker_000001");
}

TEST(StaticTrackIdUtilsTest, KeepsUnderscoreSeparatedClassNameStable) {
  EXPECT_EQ(FormatStaticTrackId("Unloading_Station", 12),
            "static_Unloading_Station_000012");
}

}  // namespace
}  // namespace automatic_annotation
