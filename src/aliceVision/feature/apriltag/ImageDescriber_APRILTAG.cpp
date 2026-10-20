// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ImageDescriber_APRILTAG.hpp"
#include <aliceVision/gpu/gpu.hpp>
#include <aliceVision/system/Logger.hpp>

#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag36h11.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace aliceVision {
namespace feature {

namespace {

/// tag36h11: number of points (corners) per tag
constexpr int tag36h11NbCorners = 4;
/// tag36h11: number of tag ids that fit in the descriptor (index = id * 4 + corner must be <= 255)
constexpr int tag36h11MaxNbIds = 64;

/**
 * @brief Override a value from an environment variable, if it is defined and valid.
 * @param[in] name The environment variable name
 * @param[in,out] value The value to override
 */
template<typename T>
void readEnvOverride(const char* name, T& value)
{
    const char* str = std::getenv(name);
    if (str == nullptr || str[0] == '\0')
        return;
    try
    {
        if constexpr (std::is_integral_v<T>)
            value = static_cast<T>(std::stoi(str));
        else
            value = static_cast<T>(std::stod(str));
    }
    catch (const std::exception& e)
    {
        ALICEVISION_LOG_WARNING("AprilTag: invalid value '" << str << "' for environment variable " << name << " (" << e.what()
                                                            << "), keep value " << value << ".");
    }
}

}  // namespace

void ImageDescriber_APRILTAG::AprilTagParameters::setPreset(EImageDescriberPreset preset) {}

ImageDescriber_APRILTAG::ImageDescriber_APRILTAG(EImageDescriberType type)
  : ImageDescriber(),
    _type(type)
{
    if (_type != EImageDescriberType::APRILTAG16H5 && _type != EImageDescriberType::APRILTAG36H11)
        throw std::invalid_argument("ImageDescriber_APRILTAG: unsupported describer type: " + EImageDescriberType_enumToString(_type));

    if (_type == EImageDescriberType::APRILTAG36H11)
    {
        readEnvOverride("ALICEVISION_APRILTAG_QUAD_DECIMATE", _quadDecimate);
        readEnvOverride("ALICEVISION_APRILTAG_MAX_HAMMING", _maxHamming);
        readEnvOverride("ALICEVISION_APRILTAG_MIN_MARGIN", _minDecisionMargin);
        ALICEVISION_LOG_INFO("AprilTag tag36h11 detector settings: quad_decimate=" << _quadDecimate << ", max hamming=" << _maxHamming
                                                                                  << ", min decision margin=" << _minDecisionMargin);
    }
}

void ImageDescriber_APRILTAG::allocate(std::unique_ptr<Regions>& regions) const
{
    if (_type == EImageDescriberType::APRILTAG36H11)
        regions.reset(new APRILTAG36H11_Regions);
    else
        regions.reset(new APRILTAG_Regions);
}

void ImageDescriber_APRILTAG::setConfigurationPreset(ConfigurationPreset preset) { _params.setPreset(preset.descPreset); }

EImageDescriberType ImageDescriber_APRILTAG::getDescriberType() const { return _type; }

bool ImageDescriber_APRILTAG::describe(const image::Image<unsigned char>& image,
                                       std::unique_ptr<Regions>& regions,
                                       const image::Image<unsigned char>* mask)
{
    // note: the mask is not used here, features are filtered by the caller (see featureEngine::FeatureExtractor)
    if (_type == EImageDescriberType::APRILTAG36H11)
        return describeTag36h11(image, regions);
    return describeTag16h5(image, regions);
}

bool ImageDescriber_APRILTAG::describeTag16h5(const image::Image<unsigned char>& image, std::unique_ptr<Regions>& regions) const
{
    allocate(regions);
    ALICEVISION_LOG_DEBUG(" regions.get() = " << regions.get());
    APRILTAG_Regions* regionsCasted = dynamic_cast<APRILTAG_Regions*>(regions.get());
    ALICEVISION_LOG_DEBUG(" regionsCasted = " << regionsCasted);

    apriltag_family_t* tf = tag16h5_create();
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;  // blur
    td->nthreads = 1;
    td->debug = 0;
    td->refine_edges = 1;

    // Make an image_u8_t header for the image
    image_u8_t im = {.width = image.width(), .height = image.height(), .stride = image.width(), .buf = (uint8_t*)image.data()};
    zarray_t* detections = apriltag_detector_detect(td, &im);
    // Draw detection outlines
    for (int i = 0; i < zarray_size(detections); i++)
    {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);
        if (det->hamming == 0)
        {
            // compute center point as the intersection of the two diagonals
            // note: AprilTag corner points are listed counter-clockwise:
            // NOTE: the variable names below do not match the actual corner positions. For a tag seen upright (as printed),
            //       p[0..3] are top-right, top-left, bottom-left, bottom-right in image coordinates (y down).
            //       The names are kept unchanged to preserve the existing tag16h5 feature layout.
            Vec2 tl(det->p[0][0], det->p[0][1]);
            Vec2 bl(det->p[1][0], det->p[1][1]);
            Vec2 br(det->p[2][0], det->p[2][1]);
            Vec2 tr(det->p[3][0], det->p[3][1]);
            double denominator = ((tl[0] - br[0]) * (bl[1] - tr[1]) - (tl[1] - br[1]) * (bl[0] - tr[0]));
            Vec2 center(((tl[0] * br[1] - tl[1] * br[0]) * (bl[0] - tr[0]) - (tl[0] - br[0]) * (bl[0] * tr[1] - bl[1] * tr[0])) / denominator,
                        ((tl[0] * br[1] - tl[1] * br[0]) * (bl[1] - tr[1]) - (tl[1] - br[1]) * (bl[0] * tr[1] - bl[1] * tr[0])) / denominator);
            Vec2 points[5] = {center, tl, bl, br, tr};
            int indices[5] = {det->id, 30 + det->id, 60 + det->id, 90 + det->id, 120 + det->id};
            // compute scale from max side length and diagonals (divided by square root of 2):
            const double scale =
              0.5 *
              std::max({(tl - bl).norm(), (bl - br).norm(), (br - tr).norm(), (tr - tl).norm(), 0.707 * (tl - br).norm(), 0.707 * (tr - bl).norm()});
            ALICEVISION_LOG_DEBUG(" New AprilTag: Id " << det->id << " ; Center location ( " << center[0] << " , " << center[1] << " ) ; Scale "
                                                       << scale);
            // ignore orientation for now:
            const float orientation = 0.0f;
            for (size_t j = 0; j < 5; ++j)
            {
                // Add its associated descriptor
                APRILTAG_Regions::DescriptorT desc;
                for (size_t k = 0; k < desc.size(); ++k)
                {
                    desc[k] = (unsigned char)0;
                }
                desc[indices[j]] = (unsigned char)255;
                regionsCasted->Descriptors().push_back(desc);
                regionsCasted->Features().push_back(PointFeature(points[j][0], points[j][1], scale, orientation));
            }
        }
    }
    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(td);
    tag16h5_destroy(tf);

    return true;
}

bool ImageDescriber_APRILTAG::describeTag36h11(const image::Image<unsigned char>& image, std::unique_ptr<Regions>& regions) const
{
    allocate(regions);
    APRILTAG36H11_Regions* regionsCasted = dynamic_cast<APRILTAG36H11_Regions*>(regions.get());
    if (regionsCasted == nullptr)
    {
        ALICEVISION_LOG_ERROR("AprilTag tag36h11: invalid regions type.");
        return false;
    }

    // The detector and the family are created for each call:
    // the same describer instance is shared between threads, so no mutable detector state is kept in the instance.
    apriltag_family_t* tf = tag36h11_create();
    apriltag_detector_t* td = apriltag_detector_create();
    // The quick decode table only needs to cover the accepted hamming distance (detections above _maxHamming are rejected anyway):
    // apriltag_detector_add_family() uses 2 bits (~36 MB table for tag36h11), 3 bits would need ~1.2 GB.
    apriltag_detector_add_family_bits(td, tf, std::clamp(_maxHamming, 0, 2));
    td->quad_decimate = _quadDecimate;  // 1.0: full resolution, as the tags are small in the frames
    td->quad_sigma = 0.0;               // blur
    td->refine_edges = 1;
    td->decode_sharpening = 0.25;
    td->nthreads = 1;
    td->debug = 0;

    // Make an image_u8_t header for the image
    image_u8_t im = {.width = image.width(), .height = image.height(), .stride = image.width(), .buf = (uint8_t*)image.data()};
    zarray_t* detections = apriltag_detector_detect(td, &im);

    // Select the valid detections
    std::vector<apriltag_detection_t*> validDetections;
    std::map<int, int> nbDetectionsPerId;
    for (int i = 0; i < zarray_size(detections); i++)
    {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        if (det->hamming > _maxHamming || det->decision_margin < _minDecisionMargin)
        {
            ALICEVISION_LOG_DEBUG(" Rejected AprilTag tag36h11: Id " << det->id << " ; hamming " << det->hamming << " ; decision margin "
                                                                     << det->decision_margin);
            continue;
        }
        if (det->id < 0 || det->id >= tag36h11MaxNbIds)
        {
            ALICEVISION_LOG_WARNING("AprilTag tag36h11: Id " << det->id << " is out of the supported range [0, " << tag36h11MaxNbIds - 1
                                                             << "], detection ignored.");
            continue;
        }
        validDetections.push_back(det);
        ++nbDetectionsPerId[det->id];
    }

    // corner k (TL, TR, BR, BL as printed) is det->p[detectionCornerIndex[k]] (see note in describeTag16h5)
    const int detectionCornerIndex[tag36h11NbCorners] = {1, 0, 3, 2};
    // there is no notion of orientation for a corner point
    const float orientation = 0.0f;

    for (const apriltag_detection_t* det : validDetections)
    {
        // the same id seen several times in one image cannot be associated without ambiguity
        if (nbDetectionsPerId.at(det->id) > 1)
        {
            ALICEVISION_LOG_WARNING("AprilTag tag36h11: Id " << det->id << " detected " << nbDetectionsPerId.at(det->id)
                                                             << " times in the same image, all of them are ignored.");
            continue;
        }

        Vec2 corners[tag36h11NbCorners];
        for (int k = 0; k < tag36h11NbCorners; ++k)
        {
            const int detCorner = detectionCornerIndex[k];
            corners[k] = Vec2(det->p[detCorner][0], det->p[detCorner][1]);
        }

        // compute scale from max side length and diagonals (divided by square root of 2):
        const double scale = 0.5 * std::max({(corners[0] - corners[1]).norm(),
                                              (corners[1] - corners[2]).norm(),
                                              (corners[2] - corners[3]).norm(),
                                              (corners[3] - corners[0]).norm(),
                                              0.707 * (corners[0] - corners[2]).norm(),
                                              0.707 * (corners[1] - corners[3]).norm()});

        ALICEVISION_LOG_DEBUG(" New AprilTag tag36h11: Id " << det->id << " ; Center location ( " << det->c[0] << " , " << det->c[1]
                                                            << " ) ; Scale " << scale << " ; hamming " << det->hamming << " ; decision margin "
                                                            << det->decision_margin);

        for (int k = 0; k < tag36h11NbCorners; ++k)
        {
            // Add its associated one-hot descriptor
            APRILTAG36H11_Regions::DescriptorT desc((unsigned char)0);
            desc[det->id * tag36h11NbCorners + k] = (unsigned char)255;
            regionsCasted->Descriptors().push_back(desc);
            regionsCasted->Features().push_back(PointFeature(corners[k][0], corners[k][1], scale, orientation));
        }
    }

    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    return true;
}

}  // namespace feature
}  // namespace aliceVision
