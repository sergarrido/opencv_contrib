/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_DICTIONARY_HPP__
#define __OPENCV_DICTIONARY_HPP__

#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>


namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{


/**
 * @brief Dictionary/Set of markers. It contains the inner codification
 *
 */
class CV_EXPORTS Dictionary {

  public:
    Mat bytesList;
    int markerSize;
    int maxCorrectionBits; // maximum number of bits that can be corrected


    /**
      */
    Dictionary(const unsigned char * bytes = 0, int _markerSize = 0, int dictsize = 0,
               int _maxcorr = 0);



    /**
     * @brief Given a matrix of bits. Returns whether if marker is identified or not.
     * It returns by reference the correct id (if any) and the correct rotation
     */
    bool identify(const Mat &onlyBits, int &idx, int &rotation, double maxCorrectionRate) const;

    /**
      * Returns the distance of the input bits to the specific id.
      */
    int getDistanceToId(InputArray bits, int id, bool allRotations = true) const;


    /**
     * @brief Draw a canonical marker image
     */
    void drawMarker(int id, int sidePixels, OutputArray _img, int borderBits = 1) const;



  private:


    /**
      * @brief Transform matrix of bits to list of bytes in the 4 rotations
      */
    Mat _getByteListFromBits(const Mat &bits) const;


    /**
      * @brief Transform list of bytes to matrix of bits
      */
    Mat _getBitsFromByteList(const Mat &byteList) const;


};




/**
 * @brief Predefined markers dictionaries/sets
 * - DICT_ARUCO: standard ArUco Library Markers. 1024 markers, 5x5 bits, 0 minimum distance
 */
enum PREDEFINED_DICTIONARY_NAME { DICT_ARUCO = 0, DICT_6X6_250 = 1 };
CV_EXPORTS const Dictionary & getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME name);



//! @}


}
}

#endif
