//
//  gravity.hpp
//  OpenTsiolkovsky
//
//  Created by Takahiro Inagawa on 2015/11/23.
//  Copyright © 2015 Takahiro Inagawa. All rights reserved.
//

#ifndef SRC_GRAVITY_HPP_
#define SRC_GRAVITY_HPP_

#include <Eigen/Core>
#include <cmath>

Eigen::Vector3d gravityECI(Eigen::Vector3d posECI_);
Eigen::Vector3d gravityECI_simple(Eigen::Vector3d posECI_);

#endif  // SRC_GRAVITY_HPP_
