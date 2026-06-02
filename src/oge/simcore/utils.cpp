//
// Created by baoyicui on 2/21/26.
//
#include "oge/simcore/utils.h"

namespace oge
{
    double stumpS(const double z)
    {
        double s;
        if (z > 0.0)
        {
            s = (sqrt(z) - sin(sqrt(z))) / pow(sqrt(z), 3);
        }
        else if (z < 0.0)
        {
            s = (sinh(sqrt(-z)) - sqrt(-z)) / pow(sqrt(-z), 3);
        }
        else
        {
            s = 1.0 / 6.0;
        }
        return s;
    }

    double stumpC(double z)
    {
        double c;
        if (z > 0.0)
        {
            c = (1.0 - cos(sqrt(z))) / z;
        }
        else if (z < 0.0)
        {
            c = (cosh(sqrt(-z)) - 1.0) / (-z);
        }
        else
        {
            c = 1.0 / 2.0;
        }
        return c;
    }

    void f_and_g(
        const double x,
        const double t,
        const double ro,
        const double a,
        double& f,
        double& g)
    {
        const double z = a * pow(x, 2);

        f = 1 - pow(x, 2) / ro * stumpC(z);
        g = t - 1 / sqrt(MU) * pow(x, 3) * stumpS(z);
    }

    void fDot_and_gDot(
        const double x,
        const double r,
        const double ro,
        const double a,
        double& fdot,
        double& gdot)
    {
        const double z = a * pow(x, 2);
        fdot = sqrt(MU) / r / ro * (z * stumpS(z) - 1) * x;
        gdot = 1 - pow(x, 2) / r * stumpC(z);
    }

    double ma2ta(double ma, double ecc, double tol, int max_iter)
    {
        // Solve Kepler's equation: E - e*sin(E) = M using Newton's method
        double E = (ecc < 0.8) ? ma : M_PI;
        for (int i = 0; i < max_iter; ++i)
        {
            double dE = (ma - E + ecc * sin(E)) / (1.0 - ecc * cos(E));
            E += dE;
            if (fabs(dE) < tol)
                break;
        }
        // Convert eccentric anomaly to true anomaly
        double ta = 2.0 * atan2(sqrt(1.0 + ecc) * sin(E / 2.0), sqrt(1.0 - ecc) * cos(E / 2.0));
        return fmod(ta + 2.0 * M_PI, 2.0 * M_PI);
    }

    void coe2rv(const Eigen::Matrix<double, 6, 1>& coe, Eigen::Vector3d& R, Eigen::Vector3d& V)
    {
        const double a = coe(0);
        const double e = coe(1);
        const double incl = coe(2);
        const double RA = coe(3);
        const double w = coe(4);
        const double TA = coe(5);

        const double h = sqrt(a * MU * (1.0 - e * e));

        Eigen::Vector3d rp = (h * h / MU) * (1.0 / (1.0 + e * cos(TA))) * (cos(TA) * Eigen::Vector3d::UnitX() +
            sin(TA) *
            Eigen::Vector3d::UnitY());
        Eigen::Vector3d vp = (MU / h) * (-sin(TA) * Eigen::Vector3d::UnitX() + (e + cos(TA)) *
            Eigen::Vector3d::UnitY());

        Eigen::Matrix3d Q_pX = (Eigen::AngleAxisd(RA, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(incl, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(w, Eigen::Vector3d::UnitZ())).toRotationMatrix();

        R = Q_pX * rp;
        V = Q_pX * vp;
    }

    void rv2coe(const Eigen::Vector3d& R, const Eigen::Vector3d& V, Eigen::Matrix<double, 6, 1>& coe)
    {
        const double eps = 1e-10;
        const double r = R.norm();
        const double v = V.norm();

        double vr = R.dot(V) / r;

        Eigen::Vector3d H = R.cross(V);
        double h = H.norm();

        double incl = (h > eps) ? std::acos(std::clamp(H(2) / h, -1.0, 1.0)) : 0.0;

        Eigen::Vector3d N = Eigen::Vector3d(0, 0, 1).cross(H);
        double n = N.norm();

        double RA = 0.0;
        if (std::abs(n) > eps)
        {
            RA = std::acos(std::clamp(N(0) / n, -1.0, 1.0));
            if (N(1) < 0)
            {
                RA = 2.0 * M_PI - RA;
            }
        }
        else
        {
            RA = 0.0;
        }

        Eigen::Vector3d E = 1.0 / MU * ((v * v - MU / r) * R - r * vr * V);
        double e = E.norm();

        double w = 0.0;
        if (std::abs(n) > eps)
        {
            if (e > eps)
            {
                w = std::acos(std::clamp(N.dot(E) / n / e, -1.0, 1.0));
                if (E(2) < 0)
                {
                    w = 2 * M_PI - w;
                }
            }
            else
            {
                w = 0.0;
            }
        }
        else
        {
            w = 0.0;
        }

        double TA = 0.0;
        if (e > eps)
        {
            TA = std::acos(std::clamp(E.dot(R) / e / r, -1.0, 1.0));
            if (vr < 0)
            {
                TA = 2 * M_PI - TA;
            }
        }
        else
        {
            if (std::abs(n) > eps)
            {
                // Circular inclined orbit: TA measured from ascending node
                Eigen::Vector3d cp = N.cross(R);
                if (cp(2) >= 0)
                {
                    TA = std::acos(std::clamp(N.dot(R) / n / r, -1.0, 1.0));
                }
                else
                {
                    TA = 2 * M_PI - std::acos(std::clamp(N.dot(R) / n / r, -1.0, 1.0));
                }
            }
            else
            {
                // Circular equatorial orbit: use true longitude (angle of R from x-axis)
                TA = std::atan2(R(1), R(0));
                if (TA < 0.0) TA += 2.0 * M_PI;
            }
        }
        double a = h * h / MU / (1 - e * e);

        coe << a, e, incl, RA, w, TA;
    }

    void DCM_J2000_to_LVLH(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM)
    {
        const Eigen::Vector3d x_hat = RRefJ2000.normalized();
        const Eigen::Vector3d z_hat = RRefJ2000.cross(VRefJ2000).normalized();
        const Eigen::Vector3d y_hat = z_hat.cross(x_hat);

        DCM.row(0) = x_hat;
        DCM.row(1) = y_hat;
        DCM.row(2) = z_hat;
    }

    void DCM_LVLH_to_J2000(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM)
    {
        Eigen::Matrix3d DCM_J2000_LVLH;
        DCM_J2000_to_LVLH(RRefJ2000, VRefJ2000, DCM_J2000_LVLH);
        DCM = DCM_J2000_LVLH.transpose();
    }

    void RV_J20002LVLH(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& RJ2000,
        const Eigen::Vector3d& VJ2000,
        Eigen::Vector3d& RLVLH,
        Eigen::Vector3d& VLVLH)
    {
        // construct LVLH frame
        Eigen::Matrix3d DCM;
        DCM_J2000_to_LVLH(RRefJ2000, VRefJ2000, DCM);
        // calculate relative position
        Eigen::Vector3d RRel = RJ2000 - RRefJ2000;
        RLVLH = DCM * RRel;
        // calculate relative velocity
        Eigen::Vector3d wJ2000 = RRefJ2000.cross(VRefJ2000) / pow(RRefJ2000.norm(), 2);
        Eigen::Vector3d VRel = (VJ2000 - VRefJ2000) - wJ2000.cross(RRel);
        VLVLH = DCM * VRel;
    }

    void RV_LVLH2J2000(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_LVLH,
        const Eigen::Vector3d& V_LVLH,
        Eigen::Vector3d& R_J2000,
        Eigen::Vector3d& V_J2000)
    {
        Eigen::Matrix3d DCM;
        DCM_LVLH_to_J2000(RRefJ2000, VRefJ2000, DCM);
        // relative position in J2000
        const Eigen::Vector3d RRel = DCM * R_LVLH;
        R_J2000 = RRel + RRefJ2000;
        // angular velocity of LVLH frame in J2000
        const Eigen::Vector3d wJ2000 = RRefJ2000.cross(VRefJ2000) / pow(RRefJ2000.norm(), 2);
        V_J2000 = DCM * V_LVLH + VRefJ2000 + wJ2000.cross(RRel);
    }

    double JulianDay(const std::chrono::sys_time<std::chrono::seconds>& UTC)
    {
        // Unix epoch (1970-01-01 00:00:00 UTC) corresponds to JD 2440587.5
        auto epoch_seconds = UTC.time_since_epoch().count();
        return static_cast<double>(epoch_seconds) / 86400.0 + 2440587.5;
    }

    double JulianDay_Modified(const std::chrono::sys_time<std::chrono::seconds>& UTC)
    {
        return JulianDay(UTC) - 2400000.5;
    }

    double UTC_TT(double MJD)
    {
        // TT (Terrestrial Time) = UTC + delta_T
        // delta_T = leap_seconds + 32.184s
        // As of 2017+, leap_seconds = 37, so delta_T ≈ 69.184s
        constexpr double delta_T_days = 69.184 / 86400.0;
        return MJD + delta_T_days;
    }

    double TT_UTC(double TT)
    {
        constexpr double delta_T_days = 69.184 / 86400.0;
        return TT - delta_T_days;
    }

    void MJD_UTC(double MJD, std::chrono::sys_time<std::chrono::seconds>& UTC)
    {
        double JD = MJD + 2400000.5;
        double seconds_since_epoch = (JD - 2440587.5) * 86400.0;
        UTC = std::chrono::sys_time<std::chrono::seconds>(
            std::chrono::seconds(static_cast<long long>(std::round(seconds_since_epoch))));
    }

    void solar_position(double jd, Eigen::Vector3d& pos_sun_j2000)
    {
        // Low-precision solar position algorithm
        // Reference: Meeus, "Astronomical Algorithms"; Vallado, "Fundamentals of Astrodynamics"

        // Julian centuries from J2000.0 epoch (2000-01-01 12:00 TT)
        double T = (jd - 2451545.0) / 36525.0;

        // Mean longitude of the Sun (degrees)
        double L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T;
        L0 = fmod(L0, 360.0);
        if (L0 < 0) L0 += 360.0;

        // Mean anomaly of the Sun (degrees)
        double M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
        M = fmod(M, 360.0);
        if (M < 0) M += 360.0;
        double M_rad = M * M_PI / 180.0;

        // Eccentricity of Earth's orbit
        double e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;

        // Equation of center (degrees)
        double C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * sin(M_rad)
            + (0.019993 - 0.000101 * T) * sin(2.0 * M_rad)
            + 0.000289 * sin(3.0 * M_rad);

        // Sun's true longitude (degrees)
        double sun_lon = L0 + C;

        // Sun's true anomaly (radians)
        double v_rad = (M + C) * M_PI / 180.0;

        // Sun-Earth distance (km)
        double R_km = AU * 1.000001018 * (1.0 - e * e) / (1.0 + e * cos(v_rad));

        // Obliquity of the ecliptic (degrees)
        double epsilon = 23.439291 - 0.0130042 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T;
        double epsilon_rad = epsilon * M_PI / 180.0;

        // Sun's ecliptic longitude (radians)
        double lambda_rad = sun_lon * M_PI / 180.0;

        // Convert from ecliptic to J2000 equatorial coordinates
        pos_sun_j2000(0) = R_km * cos(lambda_rad);
        pos_sun_j2000(1) = R_km * cos(epsilon_rad) * sin(lambda_rad);
        pos_sun_j2000(2) = R_km * sin(epsilon_rad) * sin(lambda_rad);
    }

    void solar_position(const std::chrono::sys_time<std::chrono::seconds>& UTC, Eigen::Vector3d& pos_sun_j2000)
    {
        // Convert UTC to Julian Day, then delegate
        auto epoch_seconds = UTC.time_since_epoch().count();
        double jd = static_cast<double>(epoch_seconds) / 86400.0 + 2440587.5;
        solar_position(jd, pos_sun_j2000);
    }

    double solar_illumination_angle(
        const Eigen::Vector3d& pos_sun_j2000,
        const Eigen::Vector3d& pos_evader_j2000,
        const Eigen::Vector3d& pos_chaser_j2000)
    {
        // 目标→太阳 方向
        const Eigen::Vector3d to_sun = (pos_sun_j2000 - pos_evader_j2000).normalized();
        // 目标→追击者 方向
        const Eigen::Vector3d to_chaser = (pos_chaser_j2000 - pos_evader_j2000).normalized();
        // 夹角，clamp防止数值误差导致acos越界
        return std::acos(std::clamp(to_sun.dot(to_chaser), -1.0, 1.0));
    }

    double jamming_angle(
        const Eigen::Vector3d& pos_target_j2000,
        const Eigen::Vector3d& pos_jammer_j2000)
    {
        // 目标→地心 方向（即 -pos_target 的单位向量，地心在原点）
        const Eigen::Vector3d to_earth = (-pos_target_j2000).normalized();
        // 目标→干扰星 方向
        const Eigen::Vector3d to_jammer = (pos_jammer_j2000 - pos_target_j2000).normalized();
        // 夹角，clamp防止数值误差导致acos越界
        return std::acos(std::clamp(to_earth.dot(to_jammer), -1.0, 1.0));
    }

}
