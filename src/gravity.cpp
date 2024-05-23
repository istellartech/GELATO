//
//  gravity.cpp
//  OpenTsiolkovsky
//
//  Created by Takahiro Inagawa on 2015/11/23.
//  Copyright © 2015 Takahiro Inagawa. All rights reserved.
//

#include "gravity.hpp"

Eigen::Vector3d gravityECI(Eigen::Vector3d posECI_) {
  Eigen::Vector3d gravityECI_;  // Output of this function

  /* WGS84 EGM96 */
  double a = 6378137.0;          // Equatorial radius [m]
  double one_f = 298.257223563;  // Inverse flattening [-]
  double mu = 3.986004418e14;    // Gravitational constant [m3/s2]
  double omega = 7.292115e-5;    // Angular velocity [rad/s]
  double barC20 =
      -0.484165371736e-3;  // Normalized Gravitational Coefficient \bar{C}_{20}

  double f = 1.0 / one_f;    // Flattening [-]
  double b = a * (1.0 - f);  // Polar radius [m]

  double x = posECI_[0];
  double y = posECI_[1];
  double z = posECI_[2];
  double r = std::sqrt(x * x + y * y + z * z);

  double irx, iry, irz;    // Unit position vector
  double barP20, barP20d;  // Normalized associated Legendre function
                           // \bar{P}_{20} and the derivative;
  double g_ir, g_iz;       // Gravities of r direction and z direction

  if (r == 0.0) {
    irx = iry = irz = 0;
  } else {
    irx = x / r;
    iry = y / r;
    irz = z / r;
  }

  barP20 = std::sqrt(5.0) * (3.0 * irz * irz - 1.0) * 0.5;
  barP20d = std::sqrt(5.0) * 3.0 * irz;

  if (r < b) {  // If position is under the ground
    r = b;      // Override the radius
  }

  g_ir = -mu / (r * r) *
         (1.0 + barC20 * (a / r) * (a / r) * (3.0 * barP20 + irz * barP20d));
  g_iz = mu / (r * r) * (a / r) * (a / r) * barC20 * barP20d;

  gravityECI_[0] = g_ir * irx;
  gravityECI_[1] = g_ir * iry;
  gravityECI_[2] = g_ir * irz + g_iz;
  return gravityECI_;
}

Eigen::Vector3d gravityECI_simple(Eigen::Vector3d posECI_) {
  Eigen::Vector3d gravityECI_;  // Output of this function

  /* WGS84 EGM96 */
  double mu = 3.986004418e14;    // Gravitational constant [m3/s2]

  double x = posECI_[0];
  double y = posECI_[1];
  double z = posECI_[2];
  double r = std::sqrt(x * x + y * y + z * z);


  if (r == 0.0) {
    return Eigen::Vector3d::Zero();
  } else {
    double g_ = -mu / (r * r);
    double irx = x / r;
    double iry = y / r;
    double irz = z / r;
    gravityECI_[0] = g_ * irx;
    gravityECI_[1] = g_ * iry;
    gravityECI_[2] = g_ * irz;
  }

  return gravityECI_;
}

/* void test_gravity(){
    double latitude;
    latitude = 0;
    double g = 0.0;

    std::cout << g << std::endl;
    std::ofstream ofs( "./output/gravity.csv");
    if (!ofs) {
        std::cerr << "ファイルオープンに失敗" << std::endl;
        std::exit(1);
    }
    ofs << "altitude(km)" << "\t" << "gravity(m/s2)" << endl;
    for (int altitude = 0; altitude < 1000000; altitude = altitude + 1000) {
        g = gravity(altitude, latitude);
        int alt_km = altitude / 1000;
        ofs << alt_km << "\t" << g << endl;
//        cout << g << endl;
    }
} */
