// Copyright (C) 2020  Lukasz Wandzik
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"
#include <dlib/image_io.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>


using namespace dlib;
using namespace std;

namespace py = pybind11;


// highdim_face_lbp_descriptor
std::vector<double> highdim_face_lbp_descriptor(
	    numpy_image<rgb_pixel> img,
        const full_object_detection& shape
    )
{ 
    std::vector<double> feats;
    dlib::extract_highdim_face_lbp_descriptors(img, shape, feats);
    
    return feats; 
}
// ----------------------------------------------------------------------------------------

void bind_highdim_face_lbp_descriptor(py::module &m)
{
    // highdim_face_lbp_descriptor
    m.def("highdim_face_lbp_descriptor", &highdim_face_lbp_descriptor, py::arg("img"), py::arg("shape"),
    "Takes an image and a full_object_detection that references a face in that image and converts it into a 99120D face descriptor. ");
}
