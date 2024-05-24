// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#ifndef SRC_IIP_HPP_
#define SRC_IIP_HPP_

#include <Eigen/Core>

Eigen::Vector3d posLLH_IIP_FAA(Eigen::Vector3d posECEF_,
                               Eigen::Vector3d velECEF_);

#endif  // SRC_IIP_HPP_
