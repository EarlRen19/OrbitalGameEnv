//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_UTILS_H
#define ORBITALGAMEENV_UTILS_H

#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>


#include "oge/common/constants.h"
#include "oge/simcore/propagator.h"

namespace oge
{
    /**
     * This function evaluates the Stumpff function S(z) according to Eq 3.49.
     * @param z input argument
     * @return value of S(z)
     */
    double stumpS(double z);

    /**
     * This function evaluates the Stumpff function C(z) according to Eq 3.50.
     * @param z input argument
     * @return value of C(z)
     */
    double stumpC(double z);

    /**
     *
     * @param [input] x     the universal anomaly after time t (kmˆ0.5)
     * @param [input] t     the time elapsed since t (s)
     * @param [input] ro    the radial position at time t (km)
     * @param [input] a     reciprocal of the semimajor axis (1/km)
     * @param [output] f    the Lagrange f coefficient (dimensionless)
     * @param [output] g    the Lagrange g coefficient (s)
     */
    void f_and_g(double x, double t, double ro, double a, double& f, double& g);

    void fDot_and_gDot(double x, double r, double ro, double a, double& fdot, double& gdot);

    /**
     * This function computes the state vector (r,v) from the classical orbital elements (coe).
     * @param[in] coe orbital elements [a, e, incl, RA, w, TA]
     * - a semimajor axis (km)
     * - e eccentricity
     * - incl inclination of the orbit (rad)
     * - RA right ascension of the ascending node (rad)
     * - w argument of perigee (rad)
     * - TA true anomaly (rad)
     * @param[out] R position vector in the geocentric equatorial frame (km)
     * @param[out] V velocity vector in the geocentric equatorial frame (km/s)
     */
    void coe2rv(const Eigen::Matrix<double, 6, 1>& coe, Eigen::Vector3d& R, Eigen::Vector3d& V);

    /**
     * This function computes the classical orbital elements (coe)
     * from the state vector (R,V) using Algorithm 4.1.
     * @param R position vector in the geocentric equatorial frame (km)
     * @param V velocity vector in the geocentric equatorial frame (km/s)
     * @param coe vector of orbital elements [a, e, incl, RA, w, TA]
     * - a semimajor axis (km)
     * - e eccentricity
     * - incl inclination of the orbit (rad)
     * - RA right ascension of the ascending node (rad)
     * - w argument of perigee (rad)
     * - TA true anomaly (rad))
     */
    void rv2coe(const Eigen::Vector3d& R, const Eigen::Vector3d& V, Eigen::Matrix<double, 6, 1>& coe);

    double rad2deg(double rad);
    double deg2rad(double deg);

    void DCM_LVLH_to_J2000(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM);
    void DCM_J2000_to_LVLH(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM);

    /**
     * This function transforms a position and velocity vector from the LVLH frame
     * to the J2000 inertial frame, given a reference orbit in J2000.
     * The LVLH frame is defined with x-axis along the radial direction,
     * z-axis along the orbit normal, and y-axis completing the right-handed system
     * (approximately along the velocity direction).
     * @param[in] RRefJ2000 position vector of the reference orbit in the J2000 frame (km)
     * @param[in] VRefJ2000 velocity vector of the reference orbit in the J2000 frame (km/s)
     * @param[in] R_LVLH position vector in the LVLH frame (km)
     * @param[in] V_LVLH velocity vector in the LVLH frame (km/s)
     * @param[out] R_J2000 position vector in the J2000 frame (km)
     * @param[out] V_J2000 velocity vector in the J2000 frame (km/s)
     */
    void RV_LVLH2J2000(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_LVLH,
        const Eigen::Vector3d& V_LVLH,
        Eigen::Vector3d& R_J2000,
        Eigen::Vector3d& V_J2000);

    /**
     * This function transforms a position and velocity vector from the J2000 inertial frame
     * to the LVLH frame, given a reference orbit in J2000.
     * The LVLH frame is defined with x-axis along the radial direction,
     * z-axis along the orbit normal, and y-axis completing the right-handed system
     * (approximately along the velocity direction).
     * @param[in] RRefJ2000 position vector of the reference orbit in the J2000 frame (km)
     * @param[in] VRefJ2000 velocity vector of the reference orbit in the J2000 frame (km/s)
     * @param[in] R_J2000 position vector in the J2000 frame (km)
     * @param[in] V_J2000 velocity vector in the J2000 frame (km/s)
     * @param[out] R_LVLH position vector in the LVLH frame (km)
     * @param[out] V_LVLH velocity vector in the LVLH frame (km/s)
     */
    void RV_J20002LVLH(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_J2000,
        const Eigen::Vector3d& V_J2000,
        Eigen::Vector3d& R_LVLH,
        Eigen::Vector3d& V_LVLH);

    /**
     * 将 UTC 时间转换为儒略日 (Julian Day)。
     * 利用 Unix 纪元 (1970-01-01 00:00:00 UTC) 对应 JD 2440587.5 进行计算。
     * @param[in] UTC  UTC 时间点（秒精度）
     * @return 对应的儒略日数 (JD)
     */
    double JulianDay(const std::chrono::sys_time<std::chrono::seconds>& UTC);

    /**
     * 将 UTC 时间转换为修正儒略日 (Modified Julian Day, MJD)。
     * MJD = JD - 2400000.5
     * @param[in] UTC  UTC 时间点（秒精度）
     * @return 对应的修正儒略日数 (MJD)
     */
    double JulianDay_Modified(const std::chrono::sys_time<std::chrono::seconds>& UTC);

    /**
     * 将 UTC 时间转换为地球力学时 TT (Terrestrial Time)。
     * TT = UTC + delta_T，其中 delta_T ≈ 69.184s（37 个闰秒 + 32.184s）。
     * @param[in] MJD  MJD 格式的 UTC 时间
     * @return MJD 格式的 TT 时间
     */
    double UTC_TT(double MJD);

    /**
     * 将地球力学时 TT 转换为 UTC 时间。
     * UTC = TT - delta_T，其中 delta_T ≈ 69.184s。
     * @param[in] TT  MJD 格式的 TT 时间
     * @return MJD 格式的 UTC 时间
     */
    double TT_UTC(double TT);

    /**
     * 将修正儒略日 (MJD) 转换为 UTC 时间。
     * @param[in]  MJD  修正儒略日数
     * @param[out] UTC  对应的 UTC 时间点（秒精度）
     */
    void MJD_UTC(double MJD, std::chrono::sys_time<std::chrono::seconds>& UTC);

    /**
     * 计算太阳在 J2000 坐标系下的位置向量。
     * 使用 Meeus 低精度太阳位置算法，先计算太阳黄经和日地距离，
     * 再通过黄赤交角转换到 J2000 坐标系。
     * @param[in]  jd              儒略日 (Julian Day)
     * @param[out] pos_sun_j2000   太阳在 J2000 坐标系下的位置向量 (km)
     */
    void solar_position(double jd, Eigen::Vector3d& pos_sun_j2000);

    /**
     * 计算太阳在 J2000 坐标系下的位置向量（UTC 时间输入版本）。
     * 内部将 UTC 转换为儒略日后调用 solar_position(jd, ...) 。
     * @param[in]  UTC             UTC 时间点（秒精度）
     * @param[out] pos_sun_j2000   太阳在 J2000 坐标系下的位置向量 (km)
     */
    void solar_position(const std::chrono::sys_time<std::chrono::seconds>& UTC, Eigen::Vector3d& pos_sun_j2000);

    /**
     * 计算太阳光照角：目标→太阳 与 目标→追击者 向量之间的夹角（弧度）。
     * 角度为0表示追击者在目标的顺光方向，角度越小顺光条件越好。
     * @param[in] pos_sun_j2000     太阳在 J2000 坐标系下的位置 (km)
     * @param[in] pos_evader_j2000  目标卫星在 J2000 坐标系下的位置 (km)
     * @param[in] pos_chaser_j2000  追击者卫星在 J2000 坐标系下的位置 (km)
     * @return 光照角 (rad)，范围 [0, π]
     */
    double solar_illumination_angle(
        const Eigen::Vector3d& pos_sun_j2000,
        const Eigen::Vector3d& pos_evader_j2000,
        const Eigen::Vector3d& pos_chaser_j2000);

    std::chrono::sys_time<std::chrono::milliseconds> UTC_SysTime(
        int y,
        unsigned int m,
        unsigned int d,
        int hh,
        int mm,
        double ss
    );


    class Datetime
    {
    public:
        using duration = std::chrono::milliseconds;
        using time_point = std::chrono::sys_time<duration>;

        Datetime() = default;

        Datetime(
            const int y,
            const unsigned m,
            const unsigned d,
            const int hh = 0,
            const int mm = 0,
            const int ss = 0,
            const int ms = 0
        )
            : tp_(make_time_point(y, m, d, hh, mm, ss, ms))
        {
        }

        explicit Datetime(const time_point& tp)
            : tp_(tp)
        {
        }

        static Datetime from_sys_time(const time_point& tp)
        {
            return Datetime(tp);
        }

        static Datetime from_unix_milliseconds(std::int64_t unix_ms)
        {
            return Datetime(time_point{duration{unix_ms}});
        }

        time_point to_sys_time() const
        {
            return tp_;
        }

        std::int64_t unix_milliseconds() const
        {
            return tp_.time_since_epoch().count();
        }

        int year() const
        {
            return int(ymd().year());
        }

        unsigned month() const
        {
            return unsigned(ymd().month());
        }

        unsigned day() const
        {
            return unsigned(ymd().day());
        }

        int hour() const
        {
            return int(hms().hours().count());
        }

        int minute() const
        {
            return int(hms().minutes().count());
        }

        int second() const
        {
            return int(hms().seconds().count());
        }

        int millisecond() const
        {
            return int(hms().subseconds().count());
        }

        double second_decimal() const
        {
            return static_cast<double>(second()) +
                static_cast<double>(millisecond()) / 1000.0;
        }

        std::string to_string() const
        {
            std::ostringstream oss;
            oss << std::setfill('0')
                << std::setw(4) << year() << "-"
                << std::setw(2) << month() << "-"
                << std::setw(2) << day() << " "
                << std::setw(2) << hour() << ":"
                << std::setw(2) << minute() << ":"
                << std::setw(2) << second() << "."
                << std::setw(3) << millisecond()
                << " UTC";
            return oss.str();
        }

        Datetime add_milliseconds(long long ms) const
        {
            return Datetime(tp_ + std::chrono::milliseconds{ms});
        }

        Datetime add_seconds(long long s) const
        {
            return Datetime(tp_ + std::chrono::seconds{s});
        }

        Datetime add_minutes(long long m) const
        {
            return Datetime(tp_ + std::chrono::minutes{m});
        }

        Datetime add_hours(long long h) const
        {
            return Datetime(tp_ + std::chrono::hours{h});
        }

        Datetime add_days(long long d) const
        {
            return Datetime(tp_ + std::chrono::days{d});
        }

        friend bool operator==(const Datetime&, const Datetime&) = default;

        friend bool operator<(const Datetime& a, const Datetime& b)
        {
            return a.tp_ < b.tp_;
        }

        friend bool operator<=(const Datetime& a, const Datetime& b)
        {
            return a.tp_ <= b.tp_;
        }

        friend bool operator>(const Datetime& a, const Datetime& b)
        {
            return a.tp_ > b.tp_;
        }

        friend bool operator>=(const Datetime& a, const Datetime& b)
        {
            return a.tp_ >= b.tp_;
        }

    private:
        time_point tp_{};

        static time_point make_time_point(int y, unsigned m, unsigned d,
                                          int hh, int mm, int ss, int ms)
        {
            // using namespace std::chrono::;

            std::chrono::year_month_day ymd{
                std::chrono::year{y},
                std::chrono::month{m},
                std::chrono::day{d}
            };
            if (!ymd.ok())
            {
                throw std::runtime_error("invalid date");
            }
            if (hh < 0 || hh >= 24 ||
                mm < 0 || mm >= 60 ||
                ss < 0 || ss >= 60 ||
                ms < 0 || ms >= 1000)
            {
                throw std::runtime_error("invalid time");
            }

            return std::chrono::time_point_cast<duration>(
                std::chrono::sys_days{ymd}
                + std::chrono::hours{hh}
                + std::chrono::minutes{mm}
                + std::chrono::seconds{ss}
                + std::chrono::milliseconds{ms}
            );
        }

        std::chrono::sys_days days_part() const
        {
            return std::chrono::floor<std::chrono::days>(tp_);
        }

        std::chrono::year_month_day ymd() const
        {
            return std::chrono::year_month_day{days_part()};
        }

        std::chrono::hh_mm_ss<duration> hms() const
        {
            return std::chrono::hh_mm_ss<duration>{tp_ - days_part()};
        }
    };
}

#endif //ORBITALGAMEENV_UTILS_H
