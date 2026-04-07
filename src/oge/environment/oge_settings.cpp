//
// Created by baoyicui on 3/6/26.
//

#include "oge_settings.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <sstream>

#include "oge/common/log.h"

namespace oge
{
    OGESettings::OGESettings()
    {
        // int defaults
        intSettings["random_seed"] = 42;

        // float defaults
        floatSettings["sma_base"] = 42164.0f;
        floatSettings["ecc_base"] = 0.0f;
        floatSettings["incl_base"] = 0.0f;
        floatSettings["RA_base"] = 0.0f;
        floatSettings["w_base"] = 0.0f;
        floatSettings["TA_base"] = 0.0f;

        floatSettings["dv_init_red"] = 0.2f;
        floatSettings["dv_init_blue"] = 0.2f;
        floatSettings["dv_max_per_step_red"] = 0.01f;
        floatSettings["dv_max_per_step_blue"] = 0.01f;
        floatSettings["capture_distance"] = 5.0f;
        floatSettings["timestep"] = 10.0f;
        floatSettings["terminal_time"] = 3600.0f;

        floatSettings["sma_perturb_max"] = 10.0f;
        floatSettings["dist_init_offset_min"] = 1.0f;
        floatSettings["dist_init_offset_max"] = 20.0f;
        floatSettings["jd_epoch"] = -1.0f;  // -1 = use C++ default

        floatSettings["reward_time_weight"] = -0.01; // panalty
        floatSettings["reward_formation_weight"] = 0.04;
        floatSettings["reward_fuel_weight"] = -0.05; // panalty
        floatSettings["reward_capture_weight"] = 10.0;
        floatSettings["reward_timeout_weight"] = -2.0; // panalty
        floatSettings["reward_fuelout_weight"] = -1.0; //panalty
        floatSettings["reward_phase_dist_weight"] = 1.0;

        floatSettings["reward_far_sma_penalty_scale"] = 10.0;
        floatSettings["reward_far_drift_scale"] = 5.0;
        floatSettings["reward_far_drift_max"] = 2.0;
        floatSettings["reward_far_angle_weight"] = 0.5;
        floatSettings["reward_near_energy_scale"] = 10.0;
        floatSettings["reward_near_energy_weight"] = 0.05;
        floatSettings["reward_dist_capture_bonus"] = 0.1;
        floatSettings["reward_dist_min"] = -1.0;
        floatSettings["reward_alpha_scale"] = 1.0;

        // register as internal defaults
        setInternal("random_seed", toString(intSettings["random_seed"]), -1, true);

        setInternal("sma_base", toString(floatSettings["sma_base"]), -1, true);
        setInternal("ecc_base", toString(floatSettings["ecc_base"]), -1, true);
        setInternal("incl_base", toString(floatSettings["incl_base"]), -1, true);
        setInternal("RA_base", toString(floatSettings["RA_base"]), -1, true);
        setInternal("w_base", toString(floatSettings["w_base"]), -1, true);
        setInternal("TA_base", toString(floatSettings["TA_base"]), -1, true);

        setInternal("dv_init_red", toString(floatSettings["dv_init_red"]), -1, true);
        setInternal("dv_init_blue", toString(floatSettings["dv_init_blue"]), -1, true);
        setInternal("dv_max_per_step_red", toString(floatSettings["dv_max_per_step_red"]), -1, true);
        setInternal("dv_max_per_step_blue", toString(floatSettings["dv_max_per_step_blue"]), -1, true);
        setInternal("capture_distance", toString(floatSettings["capture_distance"]), -1, true);
        setInternal("timestep", toString(floatSettings["timestep"]), -1, true);
        setInternal("terminal_time", toString(floatSettings["terminal_time"]), -1, true);

        setInternal("sma_perturb_max", toString(floatSettings["sma_perturb_max"]), -1, true);
        setInternal("dist_init_offset_min", toString(floatSettings["dist_init_offset_min"]), -1, true);
        setInternal("dist_init_offset_max", toString(floatSettings["dist_init_offset_max"]), -1, true);
        setInternal("jd_epoch", toString(floatSettings["jd_epoch"]), -1, true);

        setInternal("reward_time_weight", toString(floatSettings["reward_time_weight"]), -1, true);
        setInternal("reward_formation_weight", toString(floatSettings["reward_formation_weight"]), -1, true);
        setInternal("reward_fuel_weight", toString(floatSettings["reward_fuel_weight"]), -1, true);
        setInternal("reward_capture_weight", toString(floatSettings["reward_capture_weight"]), -1, true);
        setInternal("reward_timeout_weight", toString(floatSettings["reward_timeout_weight"]), -1, true);
        setInternal("reward_fuelout_weight", toString(floatSettings["reward_fuelout_weight"]), -1, true);
        setInternal("reward_phase_dist_weight", toString(floatSettings["reward_phase_dist_weight"]), -1, true);

        setInternal("reward_far_sma_penalty_scale", toString(floatSettings["reward_far_sma_penalty_scale"]), -1, true);
        setInternal("reward_far_drift_scale", toString(floatSettings["reward_far_drift_scale"]), -1, true);
        setInternal("reward_far_drift_max", toString(floatSettings["reward_far_drift_max"]), -1, true);
        setInternal("reward_far_angle_weight", toString(floatSettings["reward_far_angle_weight"]), -1, true);
        setInternal("reward_near_energy_scale", toString(floatSettings["reward_near_energy_scale"]), -1, true);
        setInternal("reward_near_energy_weight", toString(floatSettings["reward_near_energy_weight"]), -1, true);
        setInternal("reward_dist_capture_bonus", toString(floatSettings["reward_dist_capture_bonus"]), -1, true);
        setInternal("reward_dist_min", toString(floatSettings["reward_dist_min"]), -1, true);
        setInternal("reward_alpha_scale", toString(floatSettings["reward_alpha_scale"]), -1, true);
    }

    OGESettings::~OGESettings()
    {
        myExternalSettings.clear();
        myExternalSettings.clear();
    }

    void OGESettings::validate() const
    {
        const float sma_base = getFloat("sma_base");
        if (sma_base <= 0.0)
            throw std::invalid_argument("OGESettings: sma_base must be > 0");
        const float ecc_base = getFloat("ecc_base");
        if (ecc_base < 0.0 || ecc_base >= 1.0)
            throw std::invalid_argument("OGESettings: ecc_base must be in [0, 1)");
        const float incl_base = getFloat("incl_base");
        if (incl_base < 0.0 || incl_base > M_PI)
            throw std::invalid_argument("OGESettings: incl_base must be in [0, pi] rad");
        const float RA_base = getFloat("RA_base");
        if (RA_base < 0.0 || RA_base > 2.0 * M_PI)
            throw std::invalid_argument("OGESettings: RA_base must be in [0, 2*pi] rad");
        const float w_base = getFloat("w_base");
        if (w_base < 0.0 || w_base > 2.0 * M_PI)
            throw std::invalid_argument("OGESettings: w_base must be in [0, 2*pi] rad");
        const float TA_base = getFloat("TA_base");
        if (TA_base < 0.0 || TA_base > 2.0 * M_PI)
            throw std::invalid_argument("OGESettings: TA_base must be in [0, 2*pi] rad");

        if (getFloat("dv_init_red") <= 0.0)
            throw std::invalid_argument("OGESettings: dv_init_red must be > 0");
        if (getFloat("dv_init_blue") <= 0.0)
            throw std::invalid_argument("OGESettings: dv_init_blue must be > 0");
        if (getFloat("dv_max_per_step_red") <= 0.0)
            throw std::invalid_argument("OGESettings: dv_max_per_step_red must be > 0");
        if (getFloat("dv_max_per_step_blue") <= 0.0)
            throw std::invalid_argument("OGESettings: dv_max_per_step_blue must be > 0");
        if (getFloat("capture_distance") <= 0.0)
            throw std::invalid_argument("OGESettings: capture_distance must be > 0");
        const float timestep = getFloat("timestep");
        if (timestep <= 0.0)
            throw std::invalid_argument("OGESettings: timestep must be > 0");
        const float terminal_time = getFloat("terminal_time");
        if (terminal_time <= 0.0)
            throw std::invalid_argument("OGESettings: terminal_time must be > 0");
        if (terminal_time < timestep)
            throw std::invalid_argument("OGESettings: terminal_time must be >= timestep");

        const float sma_perturb_max = getFloat("sma_perturb_max");
        if (sma_perturb_max <= 0.0)
            throw std::invalid_argument("OGESettings: sma_perturb_max must be > 0");
        if (sma_perturb_max >= sma_base)
            throw std::invalid_argument("OGESettings: sma_perturb_max must be < sma_base");
        const float dist_init_offset_min = getFloat("dist_init_offset_min");
        const float dist_init_offset_max = getFloat("dist_init_offset_max");
        if (dist_init_offset_min < 0.0)
            throw std::invalid_argument("OGESettings: dist_init_offset_min must be >= 0");
        if (dist_init_offset_min <= 0.0)
            throw std::invalid_argument("OGESettings: dist_init_offset_min must be > 0");
        if (dist_init_offset_min >= dist_init_offset_max)
            throw std::invalid_argument("OGESettings: dist_init_offset_min must be < dist_init_offset_max");

        if (getFloat("reward_time_weight") > 0.0)
            throw std::invalid_argument("OGESettings: reward_time_weight (penalty) must be <= 0");
        if (getFloat("reward_formation_weight") < 0.0)
            throw std::invalid_argument("OGESettings: reward_formation_weight must be >= 0");
        if (getFloat("reward_fuel_weight") > 0.0)
            throw std::invalid_argument("OGESettings: reward_fuel_weight (penalty) must be <= 0");
        if (getFloat("reward_capture_weight") < 0.0)
            throw std::invalid_argument("OGESettings: reward_capture_weight must be >= 0");
        if (getFloat("reward_timeout_weight") > 0.0)
            throw std::invalid_argument("OGESettings: reward_timeout_weight (penalty) must be <= 0");
        if (getFloat("reward_fuelout_weight") > 0.0)
            throw std::invalid_argument("OGESettings: reward_fuelout_weight (penalty) must be <= 0");
        if (getFloat("reward_phase_dist_weight") < 0.0)
            throw std::invalid_argument("OGESettings: reward_phase_dist_weight must be >= 0");

    }

    void OGESettings::copyTo(OGESettings& dst) const
    {
        for (const auto& s : myInternalSettings)
            dst.setInternal(s.key, s.value);
        for (const auto& s : myExternalSettings)
            dst.setExternal(s.key, s.value);
    }


    void OGESettings::setInt(const std::string& key, const int value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(intSettings, key);
            setExternal(key, stream.str());
        }
    }


    void OGESettings::setFloat(const std::string& key, const float value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(floatSettings, key);
            setExternal(key, stream.str());
        }
    }


    void OGESettings::setBool(const std::string& key, const bool value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(boolSettings, key);
            setExternal(key, stream.str());
        }
    }


    void OGESettings::setString(const std::string& key, const std::string& value)
    {
        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, value, idx);
        }
        else
        {
            verifyVariableExistence(stringSettings, key);
            setExternal(key, value);
        }
    }


    int OGESettings::getInt(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            char* end;
            return (int)std::strtol(myInternalSettings[idx].value.c_str(), &end, 10);
        }
        else
        {
            if ((idx = getExternalPos(key)) != -1)
            {
                char* end;
                return (int)std::strtol(myExternalSettings[idx].value.c_str(), &end, 10);
            }
            else
            {
                if (strict)
                {
                    oge::Logger::Error << "No value found for key: " << key << ". ";
                    oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                    exit(-1);
                }
                else
                {
                    throw std::out_of_range("OGESettings: unknown key '" + key + "'");
                }
            }
        }
    }


    float OGESettings::getFloat(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            char* end;
            return std::strtof(myInternalSettings[idx].value.c_str(), &end);
        }
        else
        {
            if ((idx = getExternalPos(key)) != -1)
            {
                char* end;
                return std::strtof(myExternalSettings[idx].value.c_str(), &end);
            }
            else
            {
                if (strict)
                {
                    oge::Logger::Error << "No value found for key: " << key << ". ";
                    oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                    exit(-1);
                }
                else
                {
                    throw std::out_of_range("OGESettings: unknown key '" + key + "'");
                }
            }
        }
    }


    bool OGESettings::getBool(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            const std::string& value = myInternalSettings[idx].value;
            if (value == "1" || value == "true" || value == "True")
                return true;
            else if (value == "0" || value == "false" || value == "False")
                return false;
            else
                return false;
        }
        else if ((idx = getExternalPos(key)) != -1)
        {
            const std::string& value = myExternalSettings[idx].value;
            if (value == "1" || value == "true")
                return true;
            else if (value == "0" || value == "false")
                return false;
            else
                return false;
        }
        else
        {
            if (strict)
            {
                oge::Logger::Error << "No value found for key: " << key << ". ";
                oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            }
            else
            {
                throw std::out_of_range("OGESettings: unknown key '" + key + "'");
            }
        }
    }


    const std::string& OGESettings::getString(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            return myInternalSettings[idx].value;
        }
        else if ((idx = getExternalPos(key)) != -1)
        {
            return myExternalSettings[idx].value;
        }
        else
        {
            if (strict)
            {
                oge::Logger::Error << "No value found for key: " << key << ". ";
                oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            }
            else
            {
                throw std::out_of_range("OGESettings: unknown key '" + key + "'");
            }
        }
    }


    int OGESettings::getInternalPos(const std::string& key) const
    {
        for (unsigned int i = 0; i < myInternalSettings.size(); ++i)
            if (myInternalSettings[i].key == key)
                return i;

        return -1;
    }


    int OGESettings::getExternalPos(const std::string& key) const
    {
        for (unsigned int i = 0; i < myExternalSettings.size(); ++i)
            if (myExternalSettings[i].key == key)
                return i;

        return -1;
    }


    int OGESettings::setInternal(const std::string& key, const std::string& value,
                                 int pos, bool useAsInitial)
    {
        int idx = -1;

        if (pos != -1 && pos >= 0 && pos < static_cast<int>(myInternalSettings.size()) &&
            myInternalSettings[pos].key == key)
        {
            idx = pos;
        }
        else
        {
            for (unsigned int i = 0; i < myInternalSettings.size(); ++i)
            {
                if (myInternalSettings[i].key == key)
                {
                    idx = i;
                    break;
                }
            }
        }

        if (idx != -1)
        {
            myInternalSettings[idx].key = key;
            myInternalSettings[idx].value = value;
            if (useAsInitial) myInternalSettings[idx].initialValue = value;

            /*cerr << "modify internal: key = " << key
                 << ", value  = " << value
                 << ", ivalue = " << myInternalSettings[idx].initialValue
                 << " @ index = " << idx
                 << endl;*/
        }
        else
        {
            Setting setting;
            setting.key = key;
            setting.value = value;
            if (useAsInitial) setting.initialValue = value;

            myInternalSettings.push_back(setting);
            idx = myInternalSettings.size() - 1;

            /*cerr << "insert internal: key = " << key
                 << ", value  = " << value
                 << ", ivalue = " << setting.initialValue
                 << " @ index = " << idx
                 << endl;*/
        }

        return idx;
    }


    int OGESettings::setExternal(const std::string& key, const std::string& value,
                                 int pos, bool useAsInitial)
    {
        int idx = -1;

        if (pos != -1 && pos >= 0 && pos < (int)myExternalSettings.size() &&
            myExternalSettings[pos].key == key)
        {
            idx = pos;
        }
        else
        {
            for (unsigned int i = 0; i < myExternalSettings.size(); ++i)
            {
                if (myExternalSettings[i].key == key)
                {
                    idx = i;
                    break;
                }
            }
        }

        if (idx != -1)
        {
            myExternalSettings[idx].key = key;
            myExternalSettings[idx].value = value;
            if (useAsInitial) myExternalSettings[idx].initialValue = value;

            /*cerr << "modify external: key = " << key
                 << ", value = " << value
                 << " @ index = " << idx
                 << endl;*/
        }
        else
        {
            Setting setting;
            setting.key = key;
            setting.value = value;
            if (useAsInitial) setting.initialValue = value;

            myExternalSettings.push_back(setting);
            idx = myExternalSettings.size() - 1;

            /*cerr << "insert external: key = " << key
                 << ", value = " << value
                 << " @ index = " << idx
                 << endl;*/
        }

        return idx;
    }

    template <typename ValueType>
    void OGESettings::verifyVariableExistence(
        const std::map<std::string, ValueType>& dict,
        const std::string& key) const
    {
        if (dict.find(key) == dict.end())
        {
            throw std::runtime_error(
                "The key " + key + " you are trying to set does not exist or has incorrect value type.\n");
        }
    }
}
