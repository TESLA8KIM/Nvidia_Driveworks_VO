# **Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.**

@page dwx_feature_tracker_sample Feature Tracker Sample
@tableofcontents

@section dwx_feature_tracker_description Description

The Feature Tracker sample demonstrates the feature detection and feature
tracking capabilities of the @ref imageprocessing_features_mainsection module. It loads a video stream and
reads the images sequentially. For each frame, it tracks features  from the
previous frame and detects new features in the current frame.

@section dwx_feature_tracker_running Running the Sample

The feature tracker sample, sample_feature_tracker, accepts the following optional parameters. If none are specified, it will perform detections on a supplied pre-recorded video.

    ./sample_feature_tracker --video=[path/to/video.h264]
                            --maxFeatureCount=[even_number]
                            --historyCapacity=[int_number]
                            --pyramidLevel=[int_number]
                            --detectMode=[0|1]
                            --detectLevel=[int_number]
                            --cellSize=[int_number]
                            --scoreThreshold=[fp_number]
                            --detailThreshold=[fp_number]
                            --numEvenDistributionPerCell=[int_number]
                            --harrisK=[fp_number]
                            --harrisRadius=[int_number]
                            --NMSRadius=[int_number]
                            --maskType=[0|1]
                            --enableMaskAdjustment=[0|1]
                            --trackMode=[0|1]
                            --windowSize=[int_number]
                            --numIterTranslation=[int_number]
                            --numIterScaling=[int_number]
                            --numTranslationOnlyLevel=[int_number]
                            --nccUpdateThreshold=[fp_number]
                            --nccKillThreshold=[fp_number]

where

    --video=[path/to/video.h264]
        Is the absolute or relative path of a h264 or RAW/LRAW video.
        Containers such as MP4, AVI, MKV, etc. are not supported.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --maxFeatureCount=[even_number]
        Specifies the maximum number of features that can be stored.
        Note that only even numbers will work.
        Default value: 8000

    --historyCapacity=[int_number]
        Specifies how many features are going to be stored.
        Default value: 60

    --pyramidLevel=[int_number]
        Defines the number of pyramid levels while tracking.
        Higher level provides better results for large motion, but takes more time.
        Default value: 5

    --detectMode=[0|1]
        Defines detecting mode
        0: Standard Harris corner detector with fixed pipeline.
        1: Extended Harris corner detector, more flexible but takes more time.
        Default value: 0

    --cellSize=[int_number]
        Defines the size of a cell. Input images will be split into cellSize x cellSize structures.
        Default value: 64

    --scoreThreshold=[fp_number]
        Defines the minimum score for which a point is classified as a feature.
        Lower value will output more features, higher value will only keep the high frequency points.
        Default value: 4.05e-6

    --detailThreshold=[fp_number]
        Defines the minimum score for which a point is classified as a detail feature. A detail feature has higher priority to be output during detection.
        Higher values will output more even distribution, and lower values will have more features in high frequency area.
        This parameter only takes effect when --detectMode=0.
        Default value: 0.0128

    --numEvenDistributionPerCell=[int_number]
        Defines the number of features to be selected in each cell, where the score is within [scoreThreshold, detailThreshold)
        This parameter only takes effect when --detectMode=0.
        Default value: 5

    --harrisK=[fp_number]
        Defines Harris K value.
        This parameter only takes effect when --detectMode=1.
        Default value: 0.05

    --harrisRadius=[int_number]
        Defines Harris radius.
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --NMSRadius=[int_number]
        Defines non-maximum suppression filter radius.
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --maskType=[0|1]
        Defines output distribution type.
        0: Uniform distribution
        1: Gaussian distribution
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --enableMaskAdjustment=[0|1]
        Defines whether the detector will update output distribution mask before each detection.
        0: disable
        1: enable
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --trackMode=[0|1]
        Defines tracking mode.
        0: translation-only inverse additive KLT tracking.
        1: translation-scaling inverse compositional KLT tracking.
        Default value: 0

    --windowSize=[int_number]
        Defines the feature window size.
        --trackMode=0 supports windowSize=6,8,10,12,15
        --trackMode=1 supports windowSize=10,12
        Default value: 10

    --numIterTranslation=[int_number]
        Defines the KLT iteration number for translation-only tracking.
        Larger number gives better prediction, but takes more time.
        Default value: 8

    --numIterScaling=[int_number]
        Defines the KLT iteration number for translation-scaling tracking.
        Larger number gives better prediction, but takes more time.
        This parameter only takes effect when --trackMode=1.
        Default value: 10

    --numTranslationOnlyLevel=[int_number]
        Defines number of translation-only tracking in pyramid. The tracker will apply
        translation-only tracking on the highest numTranslationOnlyLevel level images.
        This parameter only takes effect when --trackMode=1.
        Default value: 4

    --nccUpdateThreshold=[fp_number]
        Defines the minimum ncc threshold that will cause the feature tracker to update
        the image template for a particular feature.
        Default value: 0.95

    --nccKillThreshold=[fp_number]
        Defines the minimum ncc threshold to mantain a particular feature in the tracker.
        Default value: 0.3 

@section dwx_feature_tracker_output Output

The sample creates a window, displays the video, and overlays the list of features.

There are two modes for feature drawing:
- (default) feature trajectories will be overlaid (up to previous 10 frames of history).
- Only the current feature positions will be overlaid by small squares.
You can switch drawing mode by pressing 'h' key.

Different colors represent the age for each feature. The age refers to how many frames have been tracked.
- Red: 1 <= age < 5
- Yellow: 5 <= age < 10
- Green: 10 <= age < 20
- Light blue: 20 <= age

![Tracked feature points on a single H.264 stream](sample_feature_tracker.png)

@section dwx_feature_tracker_more Additional Information

For more details see @ref imageprocessing_features_usecase1.
