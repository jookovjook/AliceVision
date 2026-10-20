// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ReconstructionEngine.hpp"

#include <aliceVision/config.hpp>
#include <aliceVision/system/Logger.hpp>

#include <aliceVision/feature/RegionsPerView.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>

#include <typeinfo>

namespace aliceVision {
namespace sfm {

namespace {

/**
 * @brief Decode the marker index from a one-hot marker descriptor (a single 255 value, all other values are 0).
 * @note Works for any 8-bit scalar marker regions type (CCTag, AprilTag families), using the generic Regions API.
 * @param[in] regions The marker regions of a view
 * @param[in] featureId The feature index in the regions
 * @return the marker index or UndefinedIndexT if the descriptor is not a valid one-hot marker descriptor
 */
IndexT decodeMarkerIndex(const feature::Regions& regions, std::size_t featureId)
{
    if (featureId >= regions.RegionCount())
        return UndefinedIndexT;
    if (!regions.IsScalar() || regions.Type_id() != typeid(unsigned char).name())
        return UndefinedIndexT;

    // descriptors are stored as a flat array of RegionCount() * DescriptorLength() values
    const std::size_t descLength = regions.DescriptorLength();
    const unsigned char* desc = static_cast<const unsigned char*>(regions.DescriptorRawData()) + featureId * descLength;

    IndexT markerIndex = UndefinedIndexT;
    for (std::size_t i = 0; i < descLength; ++i)
    {
        if (desc[i] == (unsigned char)255)
        {
            if (markerIndex != UndefinedIndexT)
                return UndefinedIndexT;  // more than one marker index
            markerIndex = static_cast<IndexT>(i);
        }
        else if (desc[i] != (unsigned char)0)
        {
            return UndefinedIndexT;
        }
    }
    return markerIndex;
}

}  // namespace

void retrieveMarkersId(sfmData::SfMData& sfmData, const std::vector<std::string>& extraFeaturesFolders)
{
    std::set<feature::EImageDescriberType> allMarkerDescTypes;
#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_CCTAG)
    allMarkerDescTypes.insert(feature::EImageDescriberType::CCTAG3);
    allMarkerDescTypes.insert(feature::EImageDescriberType::CCTAG4);
#endif
#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_APRILTAG)
    allMarkerDescTypes.insert(feature::EImageDescriberType::APRILTAG16H5);
    allMarkerDescTypes.insert(feature::EImageDescriberType::APRILTAG36H11);
#endif
    if (allMarkerDescTypes.empty())
        return;

    std::set<feature::EImageDescriberType> usedDescTypes = sfmData.getLandmarkDescTypes();

    std::vector<feature::EImageDescriberType> markerDescTypes;
    std::set_intersection(
      allMarkerDescTypes.begin(), allMarkerDescTypes.end(), usedDescTypes.begin(), usedDescTypes.end(), std::back_inserter(markerDescTypes));

    std::set<feature::EImageDescriberType> markerDescTypes_set(markerDescTypes.begin(), markerDescTypes.end());

    if (markerDescTypes.empty())
        return;

    // load the corresponding view regions
    feature::RegionsPerView regionPerView;
    std::set<IndexT> filter;
    // It could be optimized by loading only the minimal number of desc files,
    // but as we are only retrieving them for markers, the performance impact is limited.
    // note: the scene features folders are always used, the extra folders are added after them.
    if (!sfm::loadRegionsPerView(regionPerView, sfmData, extraFeaturesFolders, markerDescTypes, filter))
    {
        ALICEVISION_THROW_ERROR("Error while loading markers regions.");
    }

    std::size_t nbMarkerLandmarks = 0;
    std::size_t nbRetrievedMarkers = 0;
    for (auto& landmarkIt : sfmData.getLandmarks())
    {
        auto& landmark = landmarkIt.second;
        if (landmark.getObservations().empty())
            continue;
        if (markerDescTypes_set.find(landmark.getDescType()) == markerDescTypes_set.end())
            continue;
        landmark.setRgb(image::BLACK);
        ++nbMarkerLandmarks;

        // use the first observation with a valid marker descriptor
        bool found = false;
        for (const auto& obsIt : landmark.getObservations())
        {
            const feature::Regions& regions = regionPerView.getRegions(obsIt.first, landmark.getDescType());
            const IndexT markerIndex = decodeMarkerIndex(regions, obsIt.second.getFeatureId());
            if (markerIndex == UndefinedIndexT)
                continue;
            if (markerIndex > 255)
            {
                ALICEVISION_LOG_WARNING("Marker index " << markerIndex << " cannot be stored in the landmark color (landmarkId: "
                                                        << landmarkIt.first << ", viewId: " << obsIt.first << ").");
                continue;
            }
            ALICEVISION_LOG_TRACE("Found " << feature::EImageDescriberType_enumToString(landmark.getDescType()) << " marker: " << markerIndex
                                           << " (landmarkId: " << landmarkIt.first << ").");
            landmark.getRgb().r() = static_cast<unsigned char>(markerIndex);
            found = true;
            break;
        }

        if (found)
        {
            ++nbRetrievedMarkers;
        }
        else
        {
            ALICEVISION_LOG_WARNING("Failed to retrieve the marker id of the landmark "
                                    << landmarkIt.first << " (" << feature::EImageDescriberType_enumToString(landmark.getDescType()) << ").");
        }
    }
    ALICEVISION_LOG_INFO("Marker ids retrieved for " << nbRetrievedMarkers << " / " << nbMarkerLandmarks << " marker landmarks.");
}

}  // namespace sfm
}  // namespace aliceVision
