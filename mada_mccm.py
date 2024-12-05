# Import libraries for the analyses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker
import re

import scipy.stats as stats
from scipy.stats import gamma, invgamma, beta, norm, skellam
from scipy.linalg import cholesky
from scipy.stats import bootstrap
from scipy.optimize import minimize

from dataclasses import dataclass
import warnings
from google.colab import files
warnings.filterwarnings('ignore')

# Change the settings for Matplotlib to get Stata style graphs
plt.style.use("default")
plt.rc("font", family="Liberation Sans Narrow")
# Extend the ticks into the plot area
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3

# Extend the gridlines across the entire plot
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.5

# Create a formatter function to add a '$' sign as a prefix
# This formatter function will be used for Matplotlib figures
def dollar_formatter(x, pos):
    return f'${x:,.0f}'  # Adjust the format as needed

# Create a function for drawing a 95% CI ellipse around simulated values
# This function has been lifted from the Matplotlib website
def ci_ellipse(x, y, edgecolor="k"):
    """
    Draws a 95% confidence ellipse on the current matplotlib plot based on the
    given x and y data.
    Source: Matplotlib website
    Args:
        x (array-like): The x-coordinate data.
        y (array-like): The y-coordinate data.

    Returns:
        matplotlib.patches.Ellipse: The Ellipse object representing the 95% confidence ellipse.
    """
    # Calculate the mean and covariance matrix
    mean = np.array([np.mean(x), np.mean(y)])
    cov = np.cov(x, y)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Calculate the angle of rotation of the ellipse
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Calculate the width and height of the ellipse
    width = 2 * np.sqrt(eigenvalues[0]) * np.sqrt(5.991)  # 95% confidence level
    height = 2 * np.sqrt(eigenvalues[1]) * np.sqrt(5.991)  # 95% confidence level

    # Create and add the ellipse to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=np.rad2deg(angle), edgecolor=edgecolor, fc='None', lw=1)
    plt.gca().add_patch(ellipse)

    return ellipse

# We will use gamma distributions primarily for costs
# We will make adjustments to calculate alphas and betas from mean and sd
# Make a function to make this non-repetitive

def gamma_stats(mean, std_dev, size=1000):
    """
    Generate mean and 95% confidence interval for a gamma distribution based on given mean and standard deviation parameters.

    Parameters:
        mean (float): Mean of the distribution.
        std_dev (float): Standard deviation of the distribution.
        size (int): Number of samples to generate (default is 1000).

    Returns:
        tuple: Tuple containing mean and 95% confidence interval.
    """
    # Calculate shape (alpha) and scale (beta) parameters
    alpha = (mean / std_dev) ** 2
    beta = (std_dev ** 2) / mean

    # Generate gamma distribution
    simulated_data = np.random.gamma(alpha, beta, size=size)

    # Calculate mean and 95% confidence interval
    mean_value = np.mean(simulated_data)
    confidence_interval = np.percentile(simulated_data, [2.5, 97.5])

    return mean_value, confidence_interval, simulated_data, alpha, beta


def log_normal(mean_los, std_dev_los, num_samples=1000):
    """
    Simulate lognormal distributions like hospital admissions length of stay (LOS) .

    Parameters:
    mean_los (float): The mean .
    std_dev_los (float): The standard deviation.
    num_samples (int, optional): The number of samples to generate. Default is 1,000.

    Returns:
    numpy.ndarray: An array of simulated length of stay values.
    """
    # Calculate the parameters of the lognormal distribution
    mu = np.log(mean_los) - 0.5 * np.log(1 + (std_dev_los / mean_los)**2)
    sigma = np.sqrt(np.log(1 + (std_dev_los / mean_los)**2))

    # Generate random samples from the lognormal distribution
    los_samples = np.random.lognormal(mu, sigma, num_samples)

    return los_samples

def log_normal_2(mean_los, std_dev_los, num_samples=1000):
    """
    Simulate lognormal distributions like hospital admissions length of stay (LOS) .

    Parameters:
    mean_los (float): The mean .
    std_dev_los (float): The standard deviation.
    num_samples (int, optional): The number of samples to generate. Default is 1,000.

    Returns:
    numpy.ndarray: An array of simulated length of stay values.
    """
    # Calculate the parameters of the lognormal distribution
    mu = np.log(mean_los) - 0.5 * np.log(1 + (std_dev_los / mean_los)**2)
    sigma = np.sqrt(np.log(1 + (std_dev_los / mean_los)**2))

    # Generate random samples from the lognormal distribution
    los_samples = np.random.lognormal(mu, sigma, num_samples)

    return los_samples, mu, sigma


def beta_stats(mean, std_dev, num_samples=1000):
    """
    Generates random numbers from a beta distribution with a given mean and standard deviation.

    Parameters:
        mean (float): The desired mean of the beta distribution.
        std_dev (float): The desired standard deviation of the beta distribution.
        num_samples (int, optional): Number of random samples to generate. Default is 10000.

    Returns:
        numpy.ndarray: An array containing random samples from the beta distribution.
    """
    # Calculate shape parameters alpha and beta using mean and standard deviation
    alpha = ((1 - mean) / std_dev ** 2 - 1 / mean) * mean ** 2
    beta = alpha * (1 / mean - 1)

    # Generate random numbers from beta distribution
    samples = np.random.beta(alpha, beta, num_samples)
    return samples



def beta_stats_2(mean, std_dev, num_samples=1000):
    """
    Generates random numbers from a beta distribution with a given mean and standard deviation.

    Parameters:
        mean (float): The desired mean of the beta distribution.
        std_dev (float): The desired standard deviation of the beta distribution.
        num_samples (int, optional): Number of random samples to generate. Default is 10000.

    Returns:
        numpy.ndarray: An array containing random samples from the beta distribution.
    """
    # Calculate shape parameters alpha and beta using mean and standard deviation
    alpha = ((1 - mean) / std_dev ** 2 - 1 / mean) * mean ** 2
    beta = alpha * (1 / mean - 1)

    # Generate random numbers from beta distribution
    samples = np.random.beta(alpha, beta, num_samples)
    return samples, alpha, beta

#Key constants for the analyses
exchange_rate = 3905.54 # To USD as at January 2023
discount_rate = .03
median_age = 20.3
inequality_quintile = .057
ppp_ce_threshold = 133 # International dollars
lower_ppp_ce_threshold = 26 # International dollars
upper_ppp_ce_threshold = 690 # International dollars
hourly_wage_ppp = .75 ## Purchasing power parity
hourly_wage_nominal = .19
gdp_ppp = 1601 #International dollars
gdp_per_capita = 516.59
gdp_deflator_2010 = 126.679
gdp_deflator_2022 = 281.067
deflator = gdp_deflator_2022/ gdp_deflator_2010
work_week = 40 # 40 hours
inflation_official = 6.0 # Inflation rate percentage

#Demographic pyramid for drug dosage adjustment
# Dose 5-24mg/kg artemether
# 29-144mg/kg lumefantrine
# Assume 6-14 year olds
population_6_14 = 94_998
population_14_plus = 186_347
study_pop = population_6_14 + population_14_plus


# Number of Monte Carlo simulations
n_simulations = 1000

# Additional parameters
wastage_rate = 1.05 # Based on data from the field (5%)
# The wastage rate may need to be adjusted to reflect expiry
opd_costs_2010 = 0.67 # From WHO CHOICE (Without labs and drugs) per visit
ipd_costs_2010 = 2.93 # From WHO CHOICE (without labs/drugs) per day
opd_costs_2022 = opd_costs_2010 * deflator
ipd_costs_2022 = ipd_costs_2010 * deflator
cases_per_thousand = 32.04 # Malaria Atlas Project (Andres to confirm)
# Andres says this is too low
infection_prevalence = 2.75 # Per 100 children Malaria Atlas Project
mortality_rate = 9.62 # Per 100,000 Malaria Atlas Project

ipd_costs_2022

# RDT Consumable Costs
# Discuss with team if face masks and alcohol handrub are used
rdt = 1.05 * (1.1)# Per kit - PSM Assumes 10% wastage for RDTs
gloves = 0.119 * 2 * wastage_rate # Breakdown rate of 5% given in spreadsheet
lancet = 0.03995 # 200 @ $7.99 -- Assume no waste
alcohol_swabs = 0.055 * 2 * wastage_rate # Assume two per patient $5.50 for 100
safety_box = 0.0384 * 2 # We assume that a safety box is used for 50 patients. Hard to carry around.
hand_rub = 0.0262 # $2.62 for a 500 ml bottle. Assume two pump dispenser push or 2.4ml. Assume two uses per client
face_mask = 0.06 * wastage_rate #Assume current patterns. No N95. Regular surgical

total_rdt = rdt + gloves + lancet + alcohol_swabs + safety_box + hand_rub + face_mask
print(f"""The estimated total costs for RDT consumables is ${total_rdt:,.2f}.
These costs do not include waste disposal costs of ~ 15%. Adjustment done below
We assume transport costs are already included in the costs.""")
total_rdt *= 1.15
print(f"Adjusted total consumable costs is ${total_rdt:,.2f}")

# Clinician Equipment
# CHVs do not have access to BP machines under the base case
bp_machine = 40.00
thermometer = 3.65
stethoscope = 4.60
weigh_scale = 5.98
chv_equipment = .06
opd_equipment = .09

# number_of_chws = 502 # From study
# consultations = 462_440 # From Study (3 years)
# fever_cases = 422_271 # From Study (3 years)
# pos_rdt = 242_682 # From Study (3 years)
# act_mal = 228_304 # From Study (3 years)

# The values below are for one year. Remove the division below
fever_cases = 387_624
consultations = fever_cases * 462_440/422_271
pos_rdt = 375_678 # Number of RDTs done (mislabeled as pos)
act_mal = 257_889

# Based on DID results, there were an incremental 121,889 ACTs administered
# Table 3 from Andres
# Will adjust code below accordingly and model for uncertainty
inc_act = 121_889
inc_consult = consultations * inc_act/act_mal
inc_fever = fever_cases * inc_act/act_mal # Incremental fever cases seen
inc_rdt = pos_rdt * inc_act/act_mal

inc_fever/624

# Costs are in Ar. Will adjust below to USD
number_of_chws = 624
adj_factor = 624/502 # This is used for training costs
trainer_per_diem = 5_690_000
trainer_salary = 7_884_356 # Government or donors pay for the trainers wages, which should be included .
# These can be omitted if training is done by consultants and captured as such
chw_per_diem = 50_200_000
facility_incharge_per_diem = 1_500_000
refreshments = 333_000
hall_rental = 950_000
transport = 3_556_681.8
printing = 805_000

# Calculate total initial training cost
total_initial_training = (trainer_per_diem + chw_per_diem +
                          facility_incharge_per_diem +
                          refreshments + hall_rental + transport +
                          printing)/exchange_rate

total_initial_training *= adj_factor
# Assume standard deviation is 20% of the values
std_deviation = 0.2 * total_initial_training

# Run gamma simulation and return values
g_values = gamma_stats(total_initial_training, std_deviation)
mean_init = g_values[0]
print(f"Mean initial training costs: ${g_values[0]:,.2f}")
print(f"95% Confidence Interval: $[{g_values[1][0]:,.2f}, {g_values[1][1]:,.2f}]")
print("The CI are for national projection purposes. Will use actual mean for the analysis.")


# Verify if simulations work
plt.hist(g_values[2], bins=30, alpha=.7, color="b", ec="k", label="Simulated")
plt.axvline(x=np.mean(g_values[2]), color="r", linestyle="--", lw=1, label="Mean")
plt.axvline(x=g_values[1][0], color="g", linestyle="--", lw=1, label="95% CI")
plt.axvline(x=g_values[1][1], color="g", linestyle="--", lw=1)
plt.ylabel("Density")
plt.xlabel("Initial training costs ($)")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.legend();

per_diem_chv = 50_200_000
per_diem_trainer = 6_144_000
facility_incharge_per_diem = 1_500_000
refreshments = 3_625_000
hall_hire = 2_160_000
transport = 5_497_800
printing = 805_000
ref_adjustment = .8
tot_refresher = (per_diem_chv + per_diem_trainer +
                 facility_incharge_per_diem + refreshments +
                 hall_rental + printing + transport)/exchange_rate

refresher_sd = .2 * tot_refresher
ref_values = gamma_stats(tot_refresher, refresher_sd)
mean_ref = ref_values[0]
print(f"Mean refresher training costs: ${mean_ref:,.2f}")
print(f"95% Confidence Interval: $[{ref_values[1][0]:,.2f}, {ref_values[1][1]:,.2f}]")
print("The CI are for national projection purposes. Will use actual mean for the analysis.")

if mean_ref > g_values[0]:
  mean_ref *= ref_adjustment
  print(f"We will use a mean refresher value of ${mean_ref:,.2f}")


# Community sensitization costs for ICCM
# Assumes 4 days of sensitization for 15 facilities
# Assumes both intervention and control sides will be covered
sensitization = 4_873_600 * 2/ exchange_rate
sensitization *= adj_factor
print(f"Community sensitization costs were ${sensitization:,.2f}")
sensitization_per_chw = sensitization/number_of_chws
print(f"Average sensitization costs per CHW were ${sensitization_per_chw:,.2f}")


# Planning meetings (one off costs)
strategic_planning_national = 12_517_420 # (4 days for 40 participants)
district_share_national = strategic_planning_national/40
strategic_planning_district = 17_064_800 + district_share_national # (5 days for 30 participants
strategic_planning_chv = strategic_planning_district/number_of_chws
print(f"District planning costs: ${strategic_planning_district/exchange_rate:,.2f}")
print(f"CHV share of strategic planning costs per CHV ${strategic_planning_chv/exchange_rate:,.2f}")
print("=========================================================================")
print("The CHV share will be split across all patients seen")
print("We will not annualize planning costs.")

# Assume landcruiser is 60K and motorcycles 5K each
purchase_price = 70000
lifespan = 6  # in years
annuitization_rate = 0.03  # 3%

# Calculate the present value of the future cost
def calculate_present_value(future_cost, rate, years):
    return future_cost / (1 + rate) ** years

present_value = calculate_present_value(purchase_price, annuitization_rate, lifespan)
print(f"Present value of the future cost: ${present_value:.2f}")

# Calculate the annuity factor
def calculate_annuity_factor(rate, years):
    numerator = rate * (1 + rate) ** years
    denominator = (1 + rate) ** years - 1
    return numerator / denominator

annuity_factor = calculate_annuity_factor(annuitization_rate, lifespan)
print(f"Annuity factor: {annuity_factor:.4f}")

# Calculate the annualized cost
annualized_cost = present_value * annuity_factor
print(f"Annual cost for the vehicles: ${annualized_cost:.2f}")

rollout_costs = (strategic_planning_district/exchange_rate + sensitization
                 + mean_init + mean_ref)
supervision_national = 4_127.45 # USD
supervision_district = 3_057.37 # USD
annualized_cost = annualized_cost # USD per year (one vehicle ($60K) and two motorbikes ($5K each))
startup = rollout_costs + supervision_district + supervision_national + annualized_cost
print(f"The estimated program startup costs for Farangana in Y1: ${startup:,.2f}")

(mean_init + mean_ref)/startup

ip_consumables = hand_rub + gloves + face_mask
ip_consumables *= 12
#AL weighted costs
art_lume = ((.41 * population_6_14/(study_pop*2)) +
         (.82 * population_6_14/(study_pop*2)) +
         (1.78 * population_14_plus/(study_pop)))
#AQ weighted costs
aq_wt = ((.2 * population_6_14/(study_pop*2)) +
         (.3 * population_6_14/(study_pop*2)) +
         (.53 * population_14_plus/(study_pop)))

# IM artesunate (we assume severity is taken care of by proportions)
iv_artesunate = ((4.5 * population_6_14/(study_pop*2)) +
         (9.0 * population_6_14/(study_pop*2)) +
         (9.0 * population_14_plus/(study_pop)))
# Adjust artesunate for those potentially requiring more doses
iv_artesunate *= 1.3
pcm = .10

iv_giving_set = 1.0 # Assume used during entire stay
iv_fluids = .85 * 3 # Assume either 5% Dextrose or Ringers Lactate
iv_cannula = .15 # With injection vent 16/18/20/22
feso4 = 1.2 * .5 # Only half the patients get iron tablets
paracetamol = .1 # Assume 10 tablets of generic paracetamol/ibuprofen
# Blood transfusion costs
bld_trans = 120 # Based on https://pubmed.ncbi.nlm.nih.gov/26598799/
# Assume only 5% of patients get a transfusion and adjust
bld_trans *= .32 * .05
tot_ip_consumables = (ip_consumables + iv_artesunate +
                      iv_giving_set + iv_fluids + iv_cannula +
                      feso4 + art_lume + paracetamol + bld_trans)
print(f"We estimate the treatment-related consumable costs for an admitted case at: ${tot_ip_consumables:,.2f}")
print("We will conduct sensitivity analyses around this value")

# laboratory test
hemogram = 3.00 # Full blood counts
rbs = .7 # Random blood sugar
microscopy = 1.5 # Additional checks
rft = 4.00 # Renal function tests
lft = 4.00 # Liver function tests
total_ip_lab = total_rdt + hemogram + lft + rft + rbs + microscopy
print(f"We assume that total lab costs for severe malaria are: ${total_ip_lab:,.2f}")

# Generate sum of lab and consumables
ipd_cons_lab = tot_ip_consumables + total_ip_lab
print(f"We estimate total IP consumable costs at: ${ipd_cons_lab:,.2f}")

# Years lived with disability
# Number of incident cases
# Average duration of a case
# Disability weight
# Wastage is a parameter that can be adjusted to include stockouts.
# Change service utilization to reflect what the study found. For example, RDT can increase to 100% if home-based services given.
# Alternatively use mean values from the study
adherence = .808 # Adjust accordingly
wastage = .15 # Adjust accordingly
utilization = 1 # Can adjust service utilization instead and maintain efficacy levels as constant (unless multiplicative)
mal_in_fever = .3481 # proportion of malaria in cases with fever [Consult and change accordingly]
efficacy_no_prog = .748
efficacy_prog = .9195
proportion_severe = .082 # Proportion of malaria cases that are severe
cfr = 1.371e-02
protection = (100 - wastage) * utilization * adherence
cases_averted_no_prog = protection * mal_in_fever * efficacy_no_prog * proportion_severe
cases_averted_prog = protection * mal_in_fever * efficacy_prog * proportion_severe
deaths_averted_no_prog = protection * cfr * efficacy_no_prog * mal_in_fever
deaths_averted_prog = protection * cfr * efficacy_prog * mal_in_fever

deaths_averted_prog - deaths_averted_no_prog

# Estimate discounted future life years saved for each death averted
life_expectancy = 65.3 # Life expectancy at birth
avg_age_death = 2.5 # assume average age of death
discounted_future_years = np.sum([1/(1 + discount_rate)**t for t in list(range(int(life_expectancy - avg_age_death)))])
print(f"The average discounted years lived per death averted is: {discounted_future_years:,.2f} years")
print("This value will be used in estimating economic benefits of each death averted")

def chv_clinical_time(consult_time, consultations_per_year, rdt_adjustment, hourly_wage_nominal):
  tot_time = (consult_time/60) * consultations_per_year * rdt_adjustment
  avg_cost = tot_time * hourly_wage_nominal
  return tot_time, avg_cost

consult_time = 45.7 # This includes clerking, running RDT, dispensing, health promotion
consultations_per_year = consultations/(number_of_chws)
rdt_adjustment = .8 # Proportion of all reviews that get RDT

facility_travel = 6 * 12 # Assume CHV will travel to health facility once a month
report_time = 8 * 12 # Assume time for reports, drug reconciliations
tot_extra_time = facility_travel + report_time

non_clin_time_cost = hourly_wage_nominal * tot_extra_time

clinical_time, clin_time_cost = chv_clinical_time(consult_time,
                                                  consultations_per_year,
                                                  rdt_adjustment, hourly_wage_nominal)
clin_time_cost += non_clin_time_cost
donated_time = gamma_stats(clin_time_cost, clin_time_cost * .2)
average_cost = donated_time[0]
average_cost_ppp = donated_time[0] * hourly_wage_ppp/hourly_wage_nominal
lower_bound = donated_time[1][0]
upper_bound = donated_time[1][1]
lower_bound_ppp = donated_time[1][0] * hourly_wage_ppp/hourly_wage_nominal
upper_bound_ppp = donated_time[1][1] * hourly_wage_ppp/hourly_wage_nominal
print(f"Estimated Average Incremental Economic Cost of CHV time per year: ${average_cost:,.2f}. Nominal")
print(f"95% CI: $[{lower_bound:,.2f}, {upper_bound:,.2f}]")
print("=========================================================================")
print(f"Estimated Average Incremental Economic Cost of CHV time per year: ${average_cost_ppp:,.2f}. PPP")
print(f"95% CI: $[{lower_bound_ppp:,.2f}, {upper_bound_ppp:,.2f}]")
print("=========================================================================")
print("These costs do not cover travel to home, facility, or reporting time or other diseases managed.")
print("=========================================================================")

donated_time[1][1]

mean_time = np.mean(donated_time[2])
lower_time = np.quantile(donated_time[2], .025)
upper_time = np.quantile(donated_time[2], .975)
print(f"The estimated donated time by CHWs is {mean_time:,.1f} [{lower_time:,.1f}, {upper_time:,.1f}] hours per year")
print("This is time spent treating over-5s with fever")

def bootstrap_ci(data,axis):
  return np.mean(data, axis=axis)
res = bootstrap((donated_time[2],) , bootstrap_ci, confidence_level=.95,
                n_resamples=10_000, method="percentile")
ci_lower, ci_upper = res.confidence_interval
ci_lower

# Mortality benefits
# Verify with team if there are any mortality benefits
def calculate_npv(mean_age_of_death, life_expectancy, discount_rate, gdp_per_capita):
    years_of_life_lost = life_expectancy - mean_age_of_death
    npv = 0

    for year in range(1, years_of_life_lost + 1):
        discounted_value = gdp_ppp / ((1 + discount_rate) ** year)
        npv += discounted_value

    return npv

# Given parameters
median_age_of_death = round(19.3)
life_expectancy = round(67.8)
discount_rate = 0.03  # 3%
mean_age_work = 15
# Calculate NPV -- this assumes that the mean age of economic productivity is 15
# Some economists use 12 years

result_1 = calculate_npv(median_age_of_death, life_expectancy, discount_rate, gdp_ppp)
result_2 = calculate_npv(median_age_of_death, (mean_age_work - median_age_of_death), discount_rate, gdp_ppp)
print("================================================================")
print(f"Net Present Value of a life lost in Madagascar: ${result_1 - result_2:.2f}")
print("================================================================")

# Use US values as baseline
us_gdp_pc = 76_329.58 # World Bank 2022
gdp_ratio = gdp_per_capita/us_gdp_pc
eta = 1.4 # Income elasticity (will conduct sensitivity analyses around this value)
life_expectancy = 67.8
median_age = 19.3
rem_life = int(life_expectancy - median_age)
us_vsl = 11_377_977 # USDA values 2022
mada_vsl = us_vsl * (gdp_ppp/us_gdp_pc) ** eta

# Robinson et al recommend using 20 times GNI if VSL lower than GNI per capita
if mada_vsl > gdp_ppp * 20:
  pass
else:
  mada_vsl = gdp_ppp * 20
print("================================================================")
print(f"The estimated VSL for Madagascar is: ${mada_vsl:,.2f}")
print("================================================================")
# Estimate VSL per year Madagascar
vsl_year_mada = mada_vsl/np.sum([(1+discount_rate) ** -t for t in range(rem_life)])
print(f"Annualized VSL Madagascar: ${vsl_year_mada:,.2f}")
print("================================================================")
print("Note the values from the VSL approach are $10K greater than NPV.")
print("We will use the more conservative VSL figures.")

life_expectancy

#Funeral costs -- use placeholder value (to update)
funeral_cost = 1_500

# To verify numbers with Andres
# Will use these proportions to adjust the Madagascar values
malaria_cases_mean = 3_559_518
malaria_cases_lower = 2_269_000
malaria_cases_upper = 4_940_000
uncomplicated_severe_lower = .01 # World Malaria report
uncomplicated_severe_mean = .02 # Uncomplicated cases that become severe
uncomplicated_severe_upper = .03
severe_hospitalized = .65 # 50-80% of severe cases
reduction_admission_chv = .16 # IRR .84[.78-.90, p<.001] https://pubmed.ncbi.nlm.nih.gov/36927440/
reduction_mortality_chv = .22 # IRR .78[.68-.89, p<.001] https://pubmed.ncbi.nlm.nih.gov/36927440/
malaria_deaths_mean = 9_111
malaria_deaths_lower = 3_510
malaria_deaths_upper = 17_000
# Source -- World Malaria Report 2023
deaths_averted_chv = .19  # Under 5 data https://pubmed.ncbi.nlm.nih.gov/36927440/
lower_range_mal = malaria_cases_lower/malaria_cases_mean
upper_range_mal = malaria_cases_upper/malaria_cases_mean
lower_range_death = malaria_deaths_lower/malaria_deaths_mean
upper_range_death = malaria_deaths_upper/malaria_deaths_mean
# Generate standard deviations of key parameters
# These will be used in the PSA
sd_cases = ((malaria_cases_upper - malaria_cases_lower)/3.92)/malaria_cases_mean
# sd_cases will be used as a factor for getting sds

dead_sd = ((malaria_deaths_upper - malaria_deaths_lower)/3.92)/malaria_deaths_mean
severe_sd = ((uncomplicated_severe_upper - uncomplicated_severe_lower)/3.92)/uncomplicated_severe_mean
deaths_prop = malaria_deaths_mean / malaria_cases_mean
hosp_sd = (.8 - .5)/(3.92 * .65) # See above
hosp_sd

deaths_prop

# Length of stay -- convert to mean and sd
# Most papers report this as median and IQR 7 [4,10]
# S.P. Hozo, B. Djulbegovic, and I. Hozo 2005 for details
# "Estimating the Mean and Variance from the Median, Range, and the Size of a Sample," BMC Medical Research Methodology 2005, 5:13
mean_los = (4 + (2 * 7) + 10)/ 4
sd_los = np.sqrt(((4 - (2*7) + 10)/4 + (10-4)**2)/12)

@dataclass
class SimulationInputs:
  n_iterations: int = 10_000

  chv_consults: float = consultations
  chv_consults_sd: float = chv_consults * sd_cases

  chv_fever: float = fever_cases
  chv_fever_sd: float = chv_fever * sd_cases

  u_mal: float = pos_rdt
  u_mal_sd: float = u_mal * sd_cases

  # opd_averted: float = u_mal * .55 * .6 # Estimate >5 visits to OPD averted
  # opd_averted_sd: float = opd_averted * .19

  # never_opd: float = u_mal * .55 * .6 # Estimate >5 visits who never go to OPD
  # never_opd_sd: float = never_opd * .19

  # change_opd: float = u_mal * .55 * .4 * .5 # Estimate >5 visits who stop OPD (half OPD attendees)
  # change_opd_sd: float = change_opd * .19

  opd_averted: float = inc_consult # Estimate >5 visits to OPD averted
  opd_averted_sd: float = opd_averted * .19

  never_opd: float = inc_consult * .45 # Assume 45% will never go to OPD for malaria
  never_opd_sd: float = never_opd * .19

  change_opd: float = inc_consult *.55 # Assume the rest are changed
  change_opd_sd: float = change_opd * .19

  chv_act: float = inc_act
  chv_act_sd: float = chv_act * sd_cases

  severe_mal: float = opd_averted * proportion_severe
  severe_mal_sd: float = severe_mal * severe_sd

  adm_rate: float = severe_hospitalized
  adm_rate_sd: float = hosp_sd

  admissions: float = adm_rate * severe_mal
  admissions_sd: float = admissions * adm_rate_sd

  reduction_admission_chv: float = reduction_admission_chv
  reduction_admission_chv_sd: float = (.22 - .10)/(3.92 * .16)

  admissions_averted: float = reduction_admission_chv * admissions
  admissions_averted_sd: float = admissions * reduction_admission_chv_sd

  deaths: float = inc_rdt * deaths_prop
  deaths_sd: float = deaths * dead_sd

  reduction_mortality_chv: float = reduction_mortality_chv
  reduction_mortality_chv_sd: float = (.32 - .11)/(3.92 * .22)

  deaths_averted:float = deaths * reduction_mortality_chv
  deaths_averted_sd: float = deaths_averted * reduction_mortality_chv_sd

  length_of_stay: float = mean_los *1.5 # Adjust for complications like CM
  los_sd: float = sd_los * 1.2

  ip_per_diem: float = ipd_costs_2022
  ip_per_diem_sd: float = ip_per_diem * .15

  ipd_cons_lab: float = ipd_cons_lab
  ipd_cons_lab_sd: float = ipd_cons_lab * .10

  dw_um: float = .1
  dw_um_sd: float = dw_um * .19

  duration_um: float = 5.1 # Duration uncomplicated malaria
  duration_um_sd: float = duration_um * .19

  duration_sm: float = 8.75 # Duration severe
  duration_sm_comp: float = 11.0 # Duration complications

  duration_sm_overall: float = duration_sm + duration_sm_comp
  duration_sm_overall_sd: float = duration_sm_overall * .19

  dw_sm: float = .471
  dw_sm_sd: float = (.550 - .411)/(3.92 * .471)

  admission_cost: float = ip_per_diem + ipd_cons_lab
  admission_cost_sd: float = ip_per_diem_sd + (ipd_cons_lab * .19)

  startup: float = startup
  startup_sd: float = startup * .11

sim_data = SimulationInputs()

#sim_data.admissions_averted
inc_rdt

def model_simulation_inputs(sim_data):
  """
  Pick random values from different distributions
  """
  hospital_stay = log_normal(sim_data.length_of_stay, sim_data.los_sd)[0]
  admission_cost =  gamma_stats(sim_data.admission_cost, sim_data.admission_cost_sd)[2][0]
  ip_per_diem = gamma_stats(sim_data.ip_per_diem, sim_data.ip_per_diem_sd)[2][0]
  ipd_cons_lab = gamma_stats(sim_data.ipd_cons_lab, sim_data.ipd_cons_lab_sd)[2][0]
  deaths_averted = log_normal(sim_data.deaths_averted, sim_data.deaths_averted_sd)[0]
  admissions_averted = log_normal(sim_data.admissions_averted, sim_data.admissions_averted_sd)[0]
  duration_um = log_normal(sim_data.duration_um, sim_data.duration_um_sd)[0]
  duration_sm_overall = log_normal(sim_data.duration_sm_overall, sim_data.duration_sm_overall_sd)[0]
  dw_sm = beta_stats(sim_data.dw_sm, sim_data.dw_sm_sd)[0]
  dw_um = beta_stats(sim_data.dw_um, sim_data.dw_um_sd)[0]
  opd_averted = log_normal(sim_data.opd_averted, sim_data.opd_averted_sd)[0]
  never_opd = log_normal(sim_data.never_opd, sim_data.never_opd_sd)[0]
  change_opd = log_normal(sim_data.change_opd, sim_data.change_opd_sd)[0]
  startup = gamma_stats(sim_data.startup, sim_data.startup_sd)[2][0]
  return (
      hospital_stay, admission_cost,ip_per_diem ,ipd_cons_lab, deaths_averted,
      admissions_averted,
      duration_um, duration_sm_overall, dw_sm, dw_um, opd_averted,
      never_opd, change_opd, startup
  )

x = gamma_stats(183207, 45801.75)

x[3]

x[4]

#Calculate alphas and betas for the beta parameters given 95% CI and mean

# Given confidence interval and mean
ci_lower = 0.15
ci_upper = 0.34
mean = 0.26

# Define the objective function to minimize
def objective(params):
    alpha, beta_param = params
    ci_calculated_lower = beta.ppf(0.025, alpha, beta_param)
    ci_calculated_upper = beta.ppf(0.975, alpha, beta_param)
    mean_calculated = alpha / (alpha + beta_param)
    return (ci_calculated_lower - ci_lower)**2 + (ci_calculated_upper - ci_upper)**2 + (mean_calculated - mean)**2

# Initial guesses for alpha and beta
initial_guess = [1, 1]

# Minimize the objective function
result = minimize(objective, initial_guess, bounds=((0.01, None), (0.01, None)))
alpha_estimated, beta_estimated = result.x

alpha_estimated, beta_estimated


sim = model_simulation_inputs(sim_data)
sim

def model_single_run(sim_data):
  (
      hospital_stay, admission_cost, ip_per_diem, ipd_cons_lab, deaths_averted,
      admissions_averted,
      duration_um, duration_sm_overall, dw_sm, dw_um, opd_averted,
      never_opd, change_opd, startup
  ) = model_simulation_inputs(sim_data)

  return (
      hospital_stay, admission_cost, ip_per_diem, ipd_cons_lab, deaths_averted,
      admissions_averted,
      duration_um, duration_sm_overall, dw_sm, dw_um, opd_averted,
      never_opd, change_opd, startup
  )

def monte_carlo_data(sim_data):
  """
  Conduct Monte Carlo simulations with the above data
  """
  values = [model_single_run(sim_data) for i in range(sim_data.n_iterations)]
  df = pd.DataFrame(
      values,
      columns = [
          "hospital_stay", "admission_cost", "ip_per_diem","ipd_cons_lab",
          "deaths_averted", "admissions_averted",
          "duration_um", "duration_sm_overall", "dw_sm", "dw_um", "opd_averted",
          "never_opd", "change_opd", "startup"
      ]
  )
  return df

df = monte_carlo_data(sim_data)
df[:10]

df["ip_per_diem"].quantile(.025)

df["startup"].mean()

# Note that we are using the assumed change OPD and not change plus never OPD
df["opd_costs_averted"] = df['change_opd'] * (opd_costs_2022 + total_rdt +
                                              pcm + art_lume +
                            0.3 *(hemogram + microscopy))
# Ignore RDT and treatment costs for OPD costs averted to avoid double counting
df["admission_cost"] = df['ip_per_diem'] * df['hospital_stay'] + df['ipd_cons_lab']
df["admissions_costs_averted"] = df["admission_cost"] * df['admissions_averted']

# Death QALY -- based on averted deaths
df["death_qaly"] = df["deaths_averted"] * discounted_future_years
df["death_costs"] = df["deaths_averted"] * (vsl_year_mada + funeral_cost)

#DALY estimates (OPD) -- assume shorter duration of sickness (use change + never)
df["um_qaly"] = df['duration_um'] * df['dw_um'] * df['opd_averted']/365
df["sm_qaly"] = df['duration_sm_overall'] * df['dw_sm'] * df['admissions_averted']/365
df["total_qaly"] = df["death_qaly"] + df["um_qaly"] + df["sm_qaly"]
df["inc_rx"] = (df['change_opd'] + df['never_opd']) * (total_rdt + pcm + art_lume + chv_equipment) * 1.05
df["inc_prog"] = (df["inc_rx"] + df["startup"] - df['opd_costs_averted']
                  - df["admissions_costs_averted"])
etas = [i/10 for i in range(10, 26)]
for eta in etas:
  mada_vsl = us_vsl * (gdp_ppp/us_gdp_pc) ** eta
  vsl_year = mada_vsl/np.sum([(1+discount_rate) ** -t for t in range(rem_life)])
  eta = str(eta)
  df[f"death_costs_{eta}"] = df["deaths_averted"] * (vsl_year +funeral_cost)
df[:10]

# Gamma distribution parameters
# Given confidence interval and mean
ci_lower = df["death_qaly"].quantile(.025)/99.96
ci_upper = df["death_qaly"].quantile(.975)/99.96
mean = df["death_qaly"].mean()/99.96

# Define the objective function to minimize
def objective(params):
    alpha, rate = params
    ci_calculated_lower = gamma.ppf(0.025, alpha, scale=1/rate)
    ci_calculated_upper = gamma.ppf(0.975, alpha, scale=1/rate)
    mean_calculated = alpha / rate
    return (ci_calculated_lower - ci_lower)**2 + (ci_calculated_upper - ci_upper)**2 + (mean_calculated - mean)**2

# Initial guesses for alpha and rate
initial_guess = [1, 1]

# Minimize the objective function
result = minimize(objective, initial_guess, bounds=((0.01, None), (0.01, None)))
alpha_estimated, rate_estimated = result.x

alpha_estimated, rate_estimated

df["deaths_averted"].mean()

df["death_qaly"].mean()

mean_cost = np.mean(df["inc_prog"])
var_cost = np.var(df["inc_prog"], ddof=1)
mean_effect = np.mean(df["total_qaly"])
var_effect = np.var(df["total_qaly"], ddof=1)
cov_cost_effect = np.cov(df["inc_prog"], df["total_qaly"], ddof=1 )[0,1]

z = norm.ppf(.975)
a = var_cost - z**2 * var_effect
b = 2 * z**2 * mean_effect * cov_cost_effect - 2 * mean_cost * var_effect
c = z**2 * var_effect * mean_cost **2 - var_cost * mean_effect**2


discriminant = b**2 - 4 * a *c

if discriminant < 0:
  raise ValueError("The disciminant cannot be negative ")
sqrt_discriminant = np.sqrt(discriminant)
ci_lower = (b - sqrt_discriminant)/(2 * a)
ci_upper = (b + sqrt_discriminant)/(2 * a)

# Generate net health benefit -- to be used for EVPPI calculations
df["nmb_hs"] = df["total_qaly"] * 133 - df["inc_prog"]
voi = np.sum([t for t in df["nmb_hs"] if t > 0])/10000
print(f"The net monetary benefit is ${abs(voi):,.2f}")

x = df["nmb_hs"].mean()
x/np.sqrt(np.sum([(t - x)**2 for t in df["nmb_hs"] if t < 0])/10_000)

j = df["total_qaly"] * 190 - df["inc_prog"]
mean_j = j.mean()
mean_j/(np.sqrt(np.sum([(t - mean_j)**2 for t in j if t < 0])/10_000))

ceracs = []
thresholds = list(range(1,751,1))
for x in thresholds:
  j = df["total_qaly"] * x - df["inc_prog"]
  mean_j = j.mean()
  cerac = mean_j/(np.sqrt(np.sum([(t - mean_j)**2 for t in j if t < 0])/10_000))
  ceracs.append(cerac)
ceracs = [100 if x == np.inf else x for x in ceracs]
len(ceracs)

ceracs[185:190]

# Plot CERAC
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(thresholds[:240], ceracs[:240], label="CERAC")
plt.ylabel("Net benefit-to-risk ratio")
plt.xlabel("CETs in $/QALY")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(alpha=.2)
plt.axvline(133, ls="--", label="$133/QALY")
plt.legend(loc="best")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.suptitle("Cost effectiveness risk aversion curve: Madagascar MCCM")
plt.title("Health System Perspective")
plt.savefig("cerac_hs.png", bbox_inches="tight");

files.download("cerac_hs.png")

# Generate net health benefit -- to be used for EVPPI calculations
df["nhb_hs"] = df["total_qaly"] - df["inc_prog"]/133
voi = np.sum([t for t in df["nhb_hs"] if t > 0])/10000
print(f"The net health benefit is {abs(voi):,.2f} DALYs averted")

##Plot Net Monetary Benefit Curve
# Define parameters
delta_E_mean = df["total_qaly"].mean()  # Mean incremental effectiveness
delta_C_mean = df["inc_prog"].mean()  # Mean incremental cost
var_delta_E = np.var(df["total_qaly"])  # Variance of incremental effectiveness
var_delta_C = np.var(df["inc_prog"])  # Variance of incremental cost
cov_delta_E_C = np.cov(df["inc_prog"].values, df["total_qaly"].values)[0,1]  # Covariance between incremental effectiveness and cost (assumed 0 for independence)
inc_budget = df["inc_rx"].mean()
#cov_delta_E_C = 0
# Create an array of lambda values from $1000 to $100000 in increments of $1000
lambda_values = np.arange(0, 751, 1)

# Lists to store results
nmb_means = []
ci_lowers = []
ci_uppers = []
rois = []

# Calculate NMB, SE(NMB), 95% CI, and ROI for each lambda
for lambda_ in lambda_values:
    #NMB = lambda_ * delta_E_mean - delta_C_mean
    nmbs = df["total_qaly"] * lambda_ - df["inc_prog"]
    NMB = np.sum([t for t in nmbs if t > 0])/10000
    SE_NMB = np.sqrt((lambda_**2 * var_delta_E) + var_delta_C - 2 * lambda_ * cov_delta_E_C)
    CI_lower = NMB - 1.96 * SE_NMB
    CI_upper = NMB + 1.96 * SE_NMB
    # ROI = NMB / delta_C_mean

    nmb_means.append(NMB)
    ci_lowers.append(CI_lower)
    ci_uppers.append(CI_upper)
    # rois.append(ROI)

# Create a DataFrame
cov_df = pd.DataFrame({
    'Lambda': lambda_values,
    'NMB Mean': nmb_means,
    'Lower 95% CI': ci_lowers,
    'Upper 95% CI': ci_uppers,
})
cov_df["ROI"] = cov_df["NMB Mean"]/inc_budget
# Display the DataFrame
cov_df[:5]

# Optionally, save the DataFrame to a CSV file
#df.to_csv('nmb_confidence_intervals_and_roi.csv', index=False)


df["inc_rx"].quantile(.975)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(cov_df["Lambda"], cov_df['NMB Mean'], label="Mean NMB")
# plt.plot(cov_df["Lambda"], cov_df['Lower 95% CI'], label="Lower 95% CI")
# plt.plot(cov_df["Lambda"], cov_df['Upper 95% CI'], label="Upper 95% CI")
plt.fill_between(cov_df["Lambda"], cov_df['Lower 95% CI'], cov_df['Upper 95% CI'],
                 color="g", alpha=.1, zorder=-2)
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.axvline(133, ls="--", label="$133/DALY")
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.legend(bbox_to_anchor=(.9, .3))
plt.xlabel("Cost Effectiveness Threshold (WTP) in $ per QALY averted")
plt.ylabel("Net monetary benefit")
plt.suptitle("Net Monetary Benefit Expanded MCCM Madagascar")
plt.title("Health System Perspective")
plt.savefig("nmb_hs.png", bbox_inches="tight")

files.download("nmb_hs.png")

def cerac(df, cost_col, effect_col, thresholds):
  """
  This function calculates the CERAC for a given dataset and thresholds.

  Args:
      df (pandas.DataFrame): The pandas dataframe containing cost and effect data.
      cost_col (str): The column name for incremental costs.
      effect_col (str): The column name for incremental effects.
      thresholds (list): A list of cost-effectiveness thresholds.

  Returns:
      pandas.DataFrame: A dataframe containing thresholds and corresponding CERAC values.
  """
  cerac_data = []
  for threshold in thresholds:
    filtered_df = df[df[cost_col] <= threshold]
    if len(filtered_df) == 0:
      cerac_value = float('nan')  # Handle cases with no data below threshold
    else:
      downside_risk = df[df[cost_col] > threshold][cost_col].std()
      cerac_value = filtered_df[effect_col].mean() / downside_risk
    cerac_data.append([threshold, cerac_value])

  return pd.DataFrame(cerac_data, columns=["Threshold", "CERAC"])

# Assuming your data is loaded into a pandas dataframe called 'data'
cost_eff_thresholds = range(50, 1010, 1)  # Cost-effectiveness thresholds in increments of 10

cerac_df = cerac(df, "inc_prog", "total_qaly", cost_eff_thresholds)

# Plotting the CERAC curve
plt.plot(cerac_df["Threshold"], cerac_df["CERAC"])
plt.xlabel("Cost-Effectiveness Threshold ($)")
plt.ylabel("CERAC")
plt.title("Cost-Effectiveness Risk Analysis Curve (CERAC)")
plt.grid(True)
plt.show()


#Inspect distribution of simulated dataset
df.describe().round(2)

df["ip_per_diem"].quantile(.025)

df['death_costs_1.4'].quantile(.975)

icer_avg = df["inc_prog"].mean()/df["total_qaly"].mean()
icer_min = df["inc_prog"].quantile(.975)/df["total_qaly"].quantile(.025)
icer_max = df["inc_prog"].quantile(.025)/df["total_qaly"].quantile(.975)
print("=========================================================================")
print("Health System Perspective")
print(f"The incremental cost effectiveness ratio is ${icer_avg:,.2f} per DALY averted")
print(f"The 95% credibility interval for the ICER is: $[{icer_max:,.2f}, {icer_min:,.2f}] per DALY averted")
print("This ICER value should be assessed against different thresholds")
print("The intervention can be considered cost effective using conventional thresholds")
print("=========================================================================")


df["inc_prog"].min()/df["total_qaly"].max()



univ_dict = {}
for column in df.columns:
  col_mean = df[column].mean()
  col_min = df[column].quantile(.025)
  col_max = df[column].quantile(.975)
  univ_dict[column] = {
      "mean" : col_mean,
      "min" : col_min,
      "max" : col_max
  }

univ_df = pd.DataFrame(univ_dict)
univ_df[:]

# Create an empty dictionary that we will add minimum and maximum values to
univ_sens_dict = {}

def univariate_sensitivity(df, univ_df, col1, col2, col3):
    """
    Calculate the maximum and minimum incremental cost-effectiveness ratios (ICERs)
    for two different columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The main DataFrame containing the orginal data.
        univ_df (pandas.DataFrame): The DataFrame containing the column values for minimum and maximum.
        column_name1 (str): The name of the first column to use for the minimum and maximum values.
        column_name2 (str): The name of the second column to use for the minimum and maximum values.

    Returns:
        tuple: A tuple containing four values:
            - icer_max1 (float): The maximum ICER for the first column.
            - icer_min1 (float): The minimum ICER for the first column.
            - icer_max2 (float): The maximum ICER for the second column.
            - icer_min2 (float): The minimum ICER for the second column.
    """
    min_val = univ_df[col1].iloc[1]
    max_val = univ_df[col1].iloc[2]

    max_x = df[col2] * max_val
    min_x = df[col2] * min_val

    max_y = df['inc_prog'] + df[col3] - max_x
    min_y = df['inc_prog'] + df[col3] - min_x

    icer_max = np.mean(max_y) / df['total_qaly'].mean()
    icer_min = np.mean(min_y) / df['total_qaly'].mean()

    return icer_max, icer_min

#Admission costs
result = univariate_sensitivity(df, univ_df, "admission_cost", "admissions_averted", "admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Costs"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admission costs: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#Admissions averted
result = univariate_sensitivity(df, univ_df,  "admissions_averted", "admission_cost","admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Averted"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admissions averted: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#QALYs
univ_qaly_min = df['inc_prog'].mean()/df['total_qaly'].quantile(.975)
univ_qaly_max = df['inc_prog'].mean()/df['total_qaly'].quantile(.025)
univ_sens_dict["Utility"] = {"min": univ_qaly_max, "max":univ_qaly_min}
print(f"One way sensitivity analyses varying QALY: $[{univ_qaly_max:,.2f}, {univ_qaly_min:,.2f}]")

min_val = univ_df["change_opd"].iloc[1]
max_val = univ_df["change_opd"].iloc[2]
min_x = min_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_x = max_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_y = df['inc_prog'] + df['opd_costs_averted'] - min_x
min_y = df['inc_prog'] + df['opd_costs_averted'] - max_x
univ_opd_averted_max = np.mean(max_y)/df['total_qaly'].mean()
univ_opd_averted_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_opd_averted_max)
print(univ_opd_averted_min)
univ_sens_dict["OPD Costs Averted"] = {"min": univ_opd_averted_min, "max":univ_opd_averted_max}

# Inpatient consumables
mean_val = df['admission_cost'] - df['ipd_cons_lab']
min_val = mean_val + univ_df['ipd_cons_lab'][1]
max_val = mean_val + univ_df['ipd_cons_lab'][2]
min_y = df["inc_prog"] + df["admissions_costs_averted"] - (df['admissions_averted'] * max_val)
max_y = df["inc_prog"] + df["admissions_costs_averted"] - (df['admissions_averted'] * min_val)
univ_consum_max = np.mean(max_y)/df['total_qaly'].mean()
univ_consum_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_consum_min, univ_consum_max)
univ_sens_dict["IP Consumables"] = {"min": univ_consum_min, "max":univ_consum_max}


# Calculate mean value
mean_value = icer_avg
# Calculate the range for each bar and sort the dictionary based on the range
univ_sens_dict = dict(sorted(univ_sens_dict.items(), key=lambda item: item[1]['max'] - item[1]['min'], reverse=False))

# Extract labels, minimum, and maximum values from the dictionary
labels = list(univ_sens_dict.keys())
min_values = [data['min'] for data in univ_sens_dict.values()]
max_values = [data['max'] for data in univ_sens_dict.values()]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot horizontal bars
bar1 = ax.barh(labels, [mean_value - min_val for min_val in min_values], left=min_values,
        color='lightcoral', label='Minimum', height=0.4)
bar2 = ax.barh(labels, [max_val - mean_value for max_val in max_values], left=mean_value,
        color='lightblue', label='Maximum', height=0.4)

# Add mean line
ax.axvline(x=mean_value, color='gray', linestyle='--', label='Mean')

# Add labels, legend, and title
ax.set_xlabel('ICER')
ax.set_ylabel('')
fig.suptitle('Univariate Sensitivity Analyses: Health System Perspective')
ax.set_title("Madagascar Expanded MCCM Project")
ax.legend(title="ICERs")
for rect, min_val, max_val in zip(bar1.patches, min_values, max_values):
    width = rect.get_width()
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${min_val:,.0f}', ha='right', va='center', color='black')

for rect, min_val, max_val in zip(bar2.patches, min_values, max_values):
    width = rect.get_width()
    if max_val < 0:
      color = "red"
    else:
      color = "black"
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${max_val:,.0f}', ha='left', va='center', color=color)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(lw=.2, alpha=.2)
# Show plot
plt.savefig("tornado_hs_no_comp.png", bbox_inches="tight")
plt.show()

files.download("tornado_hs_no_comp.png")

# Simulate a range of potential CHV compensations per year
# Average nurse salary per year is $2770.00
# CHV salaries pegged at a maximum of 50% of this value

for i in range(1, 151):
  df[f"IC_{i}"] = (number_of_chws * i * 12) + df['inc_prog']
df[:5]

df["IC_100"].quantile(.025)/df["total_qaly"].mean()

df["IC_100"].mean()/df["total_qaly"].mean()

df["IC_100"].quantile(.975)/df["total_qaly"].mean()

threshold = 133 # CET estimated
proportions = []

for i in range(1, 101):
    column_name = f"IC_{i}"
    proportion = ((df[column_name]/df["total_qaly"])  < threshold).mean()
    proportions.append(proportion)

print(proportions)

y = [abs(x - 1) for x in proportions]
len(y)

# Select thresholds between 1 and 1000
thresholds = list(range(1,1001))
# Generate empty lists suffixed by potential compensation
probs_1 = []
probs_30 = []
probs_50 = []
probs_75 = []
probs_100 = []
probs_125 = []
probs_150 = []
# Generate a list of x values that will be used as the x-axis
x = np.arange(1,1001)
for threshold in thresholds:
  proportions = ((df["IC_1"] / df["total_qaly"]) < threshold).mean()
  probs_1.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_30"] / df["total_qaly"]) < threshold).mean()
  probs_30.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_50"] / df["total_qaly"]) < threshold).mean()
  probs_50.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_75"] / df["total_qaly"]) < threshold).mean()
  probs_75.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_100"] / df["total_qaly"]) < threshold).mean()
  probs_100.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_125"] / df["total_qaly"]) < threshold).mean()
  probs_125.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_150"] / df["total_qaly"]) < threshold).mean()
  probs_150.append(proportions)

# Plot the graphs
fig, ax = plt.subplots(figsize=(8,5), dpi=100)
plt.plot(x[:500], probs_1[:500], color="y", label="$1")
plt.plot(x[:500], probs_30[:500], color="c", label="$30")
plt.plot(x[:500], probs_50[:500], color="m", label="$50")
plt.plot(x[:500], probs_75[:500], color="k",  label="$75")
plt.plot(x[:500], probs_100[:500], color="r", ls="--", label="$100")
plt.plot(x[:500], probs_125[:500], color="b", label="$125")
plt.plot(x[:500], probs_150[:500], color="g", label="$150")
plt.grid(axis="both", lw=.3, alpha=.3)
plt.suptitle("Cost effectiveness acceptability: Health system perspective")
plt.axvline(133, ls="--", label="CET")

plt.title(" Madagascar Expanded MCCM Project")
plt.ylabel("Probability cost effective")
plt.xlabel("Threshold per DALY Averted")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.legend(loc="lower right", title="Compensation", fancybox=True, shadow=True)
ax.text(.5, -.15, "Red dotted line: Madagascar $100 NGO compensation",
        ha="center", transform=ax.transAxes,
        fontdict={'style':'italic'});
plt.savefig("ceac_hs.png", bbox_inches="tight");

files.download("ceac_hs.png")

#$1 compensation
budget = list(range(0, 300_000, 25_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_1"] < i).mean()
    proportion_below_threshold = ((df["IC_1"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames = ["${},{}".format(re.sub(r'(\d+)(\d{3})', r'\1,\2', name.split("_")[1][:-3]), name.split("_")[1][-3:]) for name in colnames]
df3.plot(figsize=(8,5))
plt.ylim(ymin=0)
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Health system perspective")
plt.title("Madagascar Expanded MCCM Project")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))

plt.text(0, -0.2, "Assume compensation of $1 per month")
plt.savefig("ceafc_hs_1.png", bbox_inches="tight")

files.download("ceafc_hs_1.png")

# $50 compensation
budget = list(range(0, 1_250_000, 100_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_50"] < i).mean()
    proportion_below_threshold = ((df["IC_50"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames = ["${},{}".format(re.sub(r'(\d+)(\d{3})', r'\1,\2', name.split("_")[1][:-3]), name.split("_")[1][-3:]) for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Health system perspective")
plt.title("Madagascar Expanded MCCM Project")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, -0.25, "Assume compensation of $50 per month")
plt.savefig("ceafc_hs_50.png", bbox_inches="tight")

files.download("ceafc_hs_50.png")

# $100 compensation
budget = list(range(0, 1_250_000, 100_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_100"] < i).mean()
    proportion_below_threshold = ((df["IC_100"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:]


colnames = df3.columns.tolist()
colnames = ["${},{}".format(re.sub(r'(\d+)(\d{3})', r'\1,\2', name.split("_")[1][:-3]), name.split("_")[1][-3:]) for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Health system perspective")
plt.title("Madagascar Expanded MCCM Project")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, -0.25, "Assume compensation of $100 per month")
plt.savefig("ceafc_hs_100.png", bbox_inches="tight")

files.download("ceafc_hs_100.png")

thresholds = [50, 100, 133, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
data = []

for threshold in thresholds:
    proportions = []
    for i in range(1, 151):
        column_name = f"IC_{i}"
        proportion = ((df[column_name] / df["total_qaly"]) < threshold).mean()
        proportions.append(proportion)
    data.append(proportions)

columns = [f"IC_{i}" for i in range(1, 151)]
result_df = pd.DataFrame(data, index=thresholds, columns=columns)
df2 = result_df.T
df2 = df2.add_prefix("CET_")
compensation = df2.index
df2 = df2.reset_index(drop=True)
#df2 = df2.reset_index(drop=False).rename(columns={"index":"Compensation"})
df2[:5]

fig, ax = plt.subplots(figsize=(8,5), dpi=100)
idx1 = (df2['CET_133'] - .8).abs().idxmin()
idx2 = (df2['CET_133'] - .5).abs().idxmin()
for col in df2.columns:
  df2[col].plot(kind="line", x=df2.index, label=col, lw=1, alpha=.7)

plt.plot(df2.index, df2['CET_133'], linestyle="--", lw=2, color="k")
plt.ylabel("Probability cost effective")
plt.xlabel("Potential monthly compensation ($)")
plt.suptitle("Potential CHW Compensation Madagascar")
plt.title("Health system perspective")
plt.legend(loc="upper right", bbox_to_anchor=(1.25,1),
           title="CE Threshold", shadow=True)
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(alpha=.3, lw=.5)
ax.text(0.5,-.15, "Dotted line: Threshold of $133 per DALY",
         ha="center", fontdict={'style':'italic'},
        transform=ax.transAxes)
# Show the plot
plt.savefig("breakeven_hs.png", bbox_inches="tight")
plt.show();


files.download("breakeven_hs.png")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_1"],s=1, color="g", alpha=.5)
l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y = df['IC_1'].values

ci_ellipse(x, y)

plt.ylabel(r'$\Delta$ Cost', fontweight="bold")  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.25, .8)
plt.xlabel(r'$\Delta$ DALYs', fontweight="bold")
ax.xaxis.set_label_coords(0.9, 0.40)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Health system perspective")
ax.text(0.6, 0, "Assume compensation of $1", ha="center",
        fontdict={'style':'italic'}, transform=ax.transAxes)
plt.legend(title="CEA Threshold", loc="best")
plt.savefig("ceplane_1_hs.png", bbox_inches="tight")
plt.show();

files.download("ceplane_1_hs.png")

above_t = np.sum([df["IC_1"]/df['total_qaly']<26])/100
print("=========================================================================")
print(f"Using a CEA threshold of $133/DALY, there is a {above_t}% probability \nthe intervention is cost effective.")
print("=========================================================================")

comp_ranges = list(range(5,155,5))
comp_ranges.insert(0, 1)
comp_str = ["$"+str(i) for i in comp_ranges]
cets = list(range(50, 700, 50))
cets.insert(0, 26)
cets.insert(3, 133)
comp_hs_df = pd.DataFrame({"Compensation": comp_str})
for j in cets:
  below_t = []
  for i in comp_ranges:
    val = str(np.round(np.sum([df[f"IC_{i}"]/df["total_qaly"]<j])/100,1))+"%"
    below_t.append(val)
  comp_hs_df[f"CET_${j}"] = below_t
comp_hs_df[:5]

comp_hs_df.to_csv("comp_hs_df.csv")
files.download("comp_hs_df.csv")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_50"],s=1, color="g", alpha=.5)
l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y = df['IC_50'].values

ci_ellipse(x, y)
plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.25, 0.80)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.40)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Health system perspective")
ax.text(0.5, 0, "Assume compensation of $50", ha="center",
        fontdict={'style':'italic'}, transform=ax.transAxes)
plt.legend(title="CEA Threshold", loc="best")

plt.show();

above_t = np.sum([df["IC_50"]/df['total_qaly']<133])/100
print("=========================================================================")
print(f"Using a CEA threshold of $300/DALY, there is a {above_t}% probability \nthe intervention is cost effective.")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_63"],s=.5, color="g", alpha=.5)

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y = df['IC_63'].values

ci_ellipse(x, y)
plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Health system perspective")
ax.text(0.5, 0, "Assume compensation of $63", ha="center",
        fontdict={'style':'italic'}, transform=ax.transAxes)
plt.legend(title="CEA Threshold", loc="best")

plt.show();



fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_63"],s=.5, color="g", alpha=.5)

# l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--", zorder=3)
# l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

plt.text(9_000, 400_000, r'$\omega_1$', color="r", fontweight="bold")
plt.axhline(375_000, color="r", ls="--", lw=1, label="Budget")
plt.text(9_000, -350_000, r'$\omega_2$', color="r", fontweight="bold")
plt.axhline(-375_000, color="r", ls="--", lw=1, )

plt.ylim(-1_000_000, 1_000_000)

plt.text(6_000, 850_000, r'$\lambda$', color="r", fontweight="bold")
plt.text(6_000, 500_000, r'$\mathit{I}$', color="r", fontweight="bold", fontsize=14)
plt.text(6_000, 250_000, r'$\mathit{II}$', color="r", fontweight="bold", fontsize=14)
plt.text(1_000, 250_000, r'$\mathit{III}$', color="r", fontweight="bold", fontsize=14)
plt.text(1_000, 500_000, r'$\mathit{IV}$', color="r", fontweight="bold", fontsize=14)

plt.ylabel(r'$\Delta$ Cost')
ax.yaxis.set_label_coords(0.25, 0.80)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.43)
plt.suptitle("CE Afffordability Expanded MCCM Madagascar")
plt.title("Health system perspective")
ax.text(0.8, 0, "Assume compensation of $63", ha="center",
        fontdict={'style':'italic'}, transform=ax.transAxes)
# plt.legend(title="CEA Threshold", loc="best")
plt.savefig("ceafc_demo.png", bbox_inches="tight")
plt.show();

files.download("ceafc_demo.png")

result = sum([1 for _, row in df.iterrows() if (row["IC_63"] / row["total_qaly"] < 133) and (row["IC_63"] < 10000)]) / 100
result

above_t = np.sum([df["IC_63"]/df['total_qaly']<133])/100
print("=========================================================================")
print(f"Using a CEA threshold of $133/DALY, there is a {above_t}% probability \nthe intervention is cost effective.")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_100"],s=1, color="g", alpha=.5)

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y = df['IC_100'].values

ci_ellipse(x, y)
plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.25, 0.80)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.40)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Assume compensation of $100")
plt.legend(title="CEA Threshold", loc="upper left")
ax.text(0.7, -0.1, "Assume compensation of $100", ha="center",
        fontdict={'style':'italic'}, transform=ax.transAxes)
plt.savefig("hs_ceplane.png", bbox_inches="tight")
plt.show();

files.download("hs_ceplane.png")

above_t = np.sum([df["IC_100"]/df['total_qaly']<133])/100
print("=========================================================================")
print(f"Using a CEA threshold of $133/DALY, there is a {above_t}% probability \nthe intervention is cost effective.")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_100"],s=.2, color="g", alpha=.5, label="$100")
sc2 = ax.scatter(df["total_qaly"], df["IC_50"] ,s=.2, color="r", alpha=.5,label="$50")
sc3 = ax.scatter(df["total_qaly"], df["IC_1"]  ,s=.2, color="b", alpha=.5,label="$1")

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_100'].values
y2 = df['IC_50'].values
y3 = df['IC_1'].values

ci_ellipse(x, y1, edgecolor="g")
ci_ellipse(x, y2, edgecolor="r")
ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Health system perspective")

#plt.legend(title="CEA Threshold", loc="upper right")
leg1 = ax.legend(handles=[sc1, sc2, sc3], loc="upper right", title="Compensation")
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="center right", title="CE Threshold")
plt.show();

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_100"],s=.2, color="g", alpha=.3, label="$100")
#sc2 = ax.scatter(df["total_qaly"], df["IC_50"] ,s=.2, color="r", alpha=.5,label="$50")
sc3 = ax.scatter(df["total_qaly"], df["IC_1"]  ,s=.2, color="r", alpha=.3,label="$1")

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_100'].values
#y2 = df['IC_50'].values
y3 = df['IC_1'].values

# ci_ellipse(x, y1, edgecolor="g")
#ci_ellipse(x, y2, edgecolor="r")
# ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.15, 0.75)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.4)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Health system perspective")

#plt.legend(title="CEA Threshold", loc="upper right")
leg1 = ax.legend(handles=[sc1, sc3], loc="upper right",
                 title="Compensation",
                 markerscale=12)
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right",
                 title="CE Threshold")
plt.savefig("cea_hs_comb.png", bbox_inches="tight")
plt.show();

files.download("cea_hs_comb.png")

df["sm_days_averted"] = df["admissions_averted"] * df['duration_sm_overall'] * df["dw_sm"]
df["um_days_averted"] = (df["change_opd"] + df["never_opd"]) * df["duration_um"] * df["dw_um"]
df["tot_um_days_averted"] = df["sm_days_averted"] + df["um_days_averted"]
df["prod_savings"] = df["tot_um_days_averted"] * hourly_wage_nominal * 8
# Assume 100% caretaker time for severe and 1 day for uncomplicated inclusive of childcare
df["caretaker_days"] = df["sm_days_averted"] + (df["change_opd"] + df["never_opd"])
df["caretaker_savings"] = df["caretaker_days"] * hourly_wage_nominal * 8
# Assume patients lose 4 hours traveling to-fro village clinic and receiving care
df["vc_loss"] = (df["change_opd"] + df["never_opd"]) * hourly_wage_nominal * 4
# Assume transport only applies to severe malaria cases. Peg at $5
df["transport"] = df["admissions_averted"] * 5
# Deduct CHW time costs -- assume 45 minutes per patient
df["chv_costs"] = hourly_wage_nominal * .75 * (df["change_opd"] + df["never_opd"])
other_chv_costs = hourly_wage_nominal * number_of_chws * 14 # 14 hours for reports and facility travel
# Estimate economic benefits without deaths
df["non_death_savings"] = (df["prod_savings"] + df["caretaker_savings"] +
                           df["transport"] - df["vc_loss"] -
                           df["chv_costs"] - other_chv_costs)

df["prod_savings"].mean()

# Simulate incremental costs from a societal perspective (no deaths)
for i in range(1, 151):
  df[f"IC_ND_{i}"] = df[f"IC_{i}"] - df["non_death_savings"]
df[:5]

# Select thresholds between 1 and 1000
thresholds = list(range(1,151))
# Generate empty lists suffixed by potential compensation
probs_1 = []
probs_30 = []
probs_50 = []
probs_75 = []
probs_100 = []
probs_125 = []
probs_150 = []
# Generate a list of x values that will be used as the x-axis
x = np.arange(1,1001)
for threshold in thresholds:
  proportions = ((df["IC_ND_1"] / df["total_qaly"]) < threshold).mean()
  probs_1.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_30"] / df["total_qaly"]) < threshold).mean()
  probs_30.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_50"] / df["total_qaly"]) < threshold).mean()
  probs_50.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_75"] / df["total_qaly"]) < threshold).mean()
  probs_75.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_100"] / df["total_qaly"]) < threshold).mean()
  probs_100.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_125"] / df["total_qaly"]) < threshold).mean()
  probs_125.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_ND_150"] / df["total_qaly"]) < threshold).mean()
  probs_150.append(proportions)

thresholds = [50, 100, 133, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
data = []

for threshold in thresholds:
    proportions = []
    for i in range(1, 151):
        column_name = f"IC_ND_{i}"
        proportion = ((df[column_name] / df["total_qaly"]) < threshold).mean()
        proportions.append(proportion)
    data.append(proportions)

columns = [f"IC_ND_{i}" for i in range(1, 151)]
result_df = pd.DataFrame(data, index=thresholds, columns=columns)
df2 = result_df.T
df2 = df2.add_prefix("CET_")
compensation = df2.index
df2 = df2.reset_index(drop=True)
df2[:5]

fig, ax = plt.subplots(figsize=(8,5))
idx1 = (df2['CET_133'] - .8).abs().idxmin()
idx2 = (df2['CET_133'] - .5).abs().idxmin()
for col in df2.columns:
  df2[col].plot(kind="line", x=df2.index, label=col, lw=1, alpha=.7)

plt.plot(df2.index, df2['CET_133'], linestyle="--", lw=2, color="k")
plt.ylabel("Probability cost effective")
plt.xlabel("Potential monthly compensation ($)")
plt.suptitle("Potential CHV Compensation Madagascar")
plt.title("Societal perspective (no mortality)")
plt.legend(loc="upper right", bbox_to_anchor=(1.25,1))
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(alpha=.3, lw=.5)
plt.text(0,-.25, "Dotted line: Threshold of $133 per DALY Averted")
# Show the plot
plt.savefig("breakeven_soc_nodeath.png", bbox_inches="tight")
plt.show();

files.download("breakeven_soc_nodeath.png")

# Generate net health benefit (no mortality)-- to be used for EVPPI calculations
df["nhb_hs"] = df["total_qaly"] - df["IC_ND_1"]/133
voi = np.sum([t for t in df["nhb_hs"] if t > 0])/10000
print(f"The net health benefit is {abs(voi):,.2f} DALYs averted")

# Generate net health benefit -- to be used for EVPPI calculations
df["nmb_hs"] = df["total_qaly"] * 133 - df["IC_ND_1"]
voi = np.sum([t for t in df["nmb_hs"] if t > 0])/10000
print(f"The net monetary benefit is ${abs(voi):,.2f}")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_ND_100"],s=.02, color="g", label="$100")
plt.scatter(df["total_qaly"], df["IC_ND_46"] ,s=.02, color="r", label="$46")
plt.scatter(df["total_qaly"], df["IC_ND_1"]  ,s=.02, color="b", label="$1")

plt.plot([-5_000, 12_000], [-150_000, 360_000], label="$30/DALY", ls="--")
plt.plot([-5_000, 10_000], [-1_500_000, 3_000_000], label="$300/DALY", ls="--")
plt.plot([-5_000, 6_000], [-3_500_000, 4_200_000], label="$700/DALY", ls="--")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_ND_100'].values
y2 = df['IC_ND_46'].values
y3 = df['IC_ND_1'].values

ci_ellipse(x, y1)
ci_ellipse(x, y2)
ci_ellipse(x, y3)

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (No mortality)")
plt.legend(title="CEA Threshold", loc="upper right", markerscale=12)

plt.show();

above_t = np.sum([df["IC_ND_100"]/df['total_qaly']<133])/100
print("=========================================================================")
print(f"Using a CEA threshold of $133/DALY, there is a {above_t}% probability \nthe intervention is cost effective.")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_ND_100"],s=.2, color="g", alpha=.3, label="$100")
# plt.scatter(df["total_qaly"], df["IC_ND_46"] ,s=.02, color="r", label="$46")
sc3 = ax.scatter(df["total_qaly"], df["IC_ND_1"]  ,s=.2, color="r", alpha=.3, label="$1")

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel("")

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_ND_100'].values
# y2 = df['IC_ND_46'].values
y3 = df['IC_ND_1'].values

# ci_ellipse(x, y1)
# ci_ellipse(x, y2)
# ci_ellipse(x, y3)

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.15, 0.8)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.4)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (No mortality)")
# plt.legend(title="CEA Threshold", loc="lower right", markerscale=12)
#plt.legend(title="CEA Threshold", loc="upper right")
leg1 = ax.legend(handles=[sc1, sc3], loc="upper right",
                 title="Compensation",
                 markerscale=12)
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right",
                 title="CE Threshold")
plt.savefig("cep_soc_nodeath.png", bbox_inches="tight")
plt.show();

files.download("cep_soc_nodeath.png")

# $1 compensation for Societal Perspective (No deaths)
budget = list(range(0, 300_000, 15_000))
thresholds = list(range(0,200,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_ND_1"] < i).mean()
    proportion_below_threshold = ((df["IC_ND_1"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded MCCM Project: No Mortality")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, 0.991, "Assume compensation of $1 per month")
plt.savefig("ceafc_soc_nd_1.png", bbox_inches="tight")

files.download("ceafc_soc_nd_1.png")

# $50 compensation for Societal Perspective (No deaths)
budget = list(range(0, 700_000, 50_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_ND_50"] < i).mean()
    proportion_below_threshold = ((df["IC_ND_50"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded MCCM Project: No Mortality")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, -0.05, "Assume compensation of $50 per month")
plt.savefig("ceafc_soc_nd_50.png", bbox_inches="tight")

files.download("ceafc_soc_nd_50.png")

# $100 compensation for Societal Perspective (No deaths)
budget = list(range(0, 1_250_000, 100_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_ND_100"] < i).mean()
    proportion_below_threshold = ((df["IC_ND_100"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded MCCM Project: No Mortality")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, -0.25, "Assume compensation of $100 per month")
plt.savefig("ceafc_soc_nd_100.png", bbox_inches="tight")

files.download("ceafc_soc_nd_100.png")

# Include mortality benefits
# Simulate incremental costs from a societal perspective (no deaths)
for i in range(1, 151):
  df[f"IC_D_{i}"] = df[f"IC_{i}"] - df["non_death_savings"] - df['death_costs']
df[["IC_ND_100", "IC_100", "IC_D_100"]][:5]

# Generate net health benefit -- to be used for EVPPI calculations
df["nmb_hs"] = df["total_qaly"] * 133 - df["IC_D_1"]
voi = np.sum([t for t in df["nmb_hs"] if t > 0])/10000
print(f"The net monetary benefit is ${abs(voi):,.2f}")

result = sum([1 for _, row in df.iterrows() if (row["IC_1"] / row["total_qaly"] < 133) and (row["IC_1"] < -1000)]) / 100
result

thresholds = [50, 100, 133, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
data = []

for threshold in thresholds:
    proportions = []
    for i in range(1, 151):
        column_name = f"IC_D_{i}"
        proportion = ((df[column_name] / df["total_qaly"]) < threshold).mean()
        proportions.append(proportion)
    data.append(proportions)

columns = [f"IC_D_{i}" for i in range(1, 151)]
result_df = pd.DataFrame(data, index=thresholds, columns=columns)
df2 = result_df.T
df2 = df2.add_prefix("CET_")
compensation = df2.index
df2 = df2.reset_index(drop=True)
df2[:5]

# Select thresholds between 1 and 1000
thresholds = list(range(1,1001))
# Generate empty lists suffixed by potential compensation
probs_1 = []
probs_30 = []
probs_50 = []
probs_75 = []
probs_100 = []
probs_125 = []
probs_150 = []
# Generate a list of x values that will be used as the x-axis
x = np.arange(1,1001)
for threshold in thresholds:
  proportions = ((df["IC_D_1"] / df["total_qaly"]) < threshold).mean()
  probs_1.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_30"] / df["total_qaly"]) < threshold).mean()
  probs_30.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_50"] / df["total_qaly"]) < threshold).mean()
  probs_50.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_75"] / df["total_qaly"]) < threshold).mean()
  probs_75.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_100"] / df["total_qaly"]) < threshold).mean()
  probs_100.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_125"] / df["total_qaly"]) < threshold).mean()
  probs_125.append(proportions)
for threshold in thresholds:
  proportions = ((df["IC_D_150"] / df["total_qaly"]) < threshold).mean()
  probs_150.append(proportions)

# Plot the graphs
fig, ax = plt.subplots(figsize=(8,5), dpi=100)
plt.plot(x[:500], probs_1[:500], color="b", label="$1")
plt.plot(x[:500], probs_50[:500], color="y", label="$50")
plt.plot(x[:500], probs_75[:500], color="k", label="$75")
plt.plot(x[:500], probs_100[:500], color="r", ls="--", label="$100")
plt.plot(x[:500], probs_125[:500], color="brown", label="$125")
plt.plot(x[:500], probs_150[:500], color="g", label="$150")
plt.grid(axis="both", lw=.3, alpha=.3)
plt.suptitle("Cost effectiveness acceptability Madagascar Expanded MCCM")
plt.title("Societal perspective: Mortality Impacts")
plt.ylabel("Probability cost effective")
plt.xlabel("Threshold per DALY Averted")
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.axvline(133, label="CET", ls="--")
plt.legend(loc="lower right", title="Compensation", fancybox=True, shadow=True)
# ax.text(.5, -.15, "Red dotted line: Madagascar trial compensation",
#         ha="center", transform=ax.transAxes,
#         fontdict={'style':'italic'})
plt.savefig("ceac_soc_nodeath.png", bbox_inches="tight")

files.download("ceac_soc_nodeath.png")

fig, ax = plt.subplots(figsize=(8,5), dpi=100)
idx1 = (df2['CET_133'] - .8).abs().idxmin()
idx2 = (df2['CET_133'] - .5).abs().idxmin()
for col in df2.columns:
  df2[col].plot(kind="line", x=df2.index, label=col, lw=1, alpha=.7)

plt.plot(df2.index, df2['CET_133'], linestyle="--", lw=2, color="k")
plt.ylabel("Probability cost effective")
plt.xlabel("Potential monthly compensation ($)")
plt.suptitle("Potential CHW Compensation Madagascar")
plt.title("Societal perspective with mortality benefits")
plt.legend(loc="upper right", bbox_to_anchor=(1.25,1))
# Apply the formatter to the x-axis
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(alpha=.3, lw=.5)
plt.text(0,-.05, "Dotted line: Threshold of $133 per DALY Averted")
# Show the plot
plt.savefig("soc_mort_comp.png", bbox_inches="tight")
plt.show();

files.download("soc_mort_comp.png")

fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(df["total_qaly"], df["IC_D_1"]  ,s=.2, color="b", alpha=.5, label="$1")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))


# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values

y3 = df['IC_D_1'].values


ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (mortality benefits included)")
plt.legend(title="CEA Threshold", loc="upper right", markerscale=12)

plt.show();

above_300 = np.sum([df["IC_D_1"]/df['total_qaly']<133])/100
above_30 = np.sum([df["IC_D_1"]/df['total_qaly']<30])/100
print("=========================================================================")
print(f"Using a CEA threshold of $300/DALY averted, there is a {above_300}% probability \nthe intervention is cost effective.")
print("")
print(f"Using a CEA threshold of $30/DALY averted, there is a {above_30}% probability \nthe intervention is cost effective.")
print("A CEA threshold of $30/DALY averted is quite conservative")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(df["total_qaly"], df["IC_D_50"]  ,s=.2, color="b", alpha=.5, label="$50")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values

y3 = df['IC_D_50'].values

ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (mortality benefits included)")
plt.legend(title="CEA Threshold", loc="upper right", markerscale = 12)

plt.show();

above_300 = np.sum([df["IC_D_50"]/df['total_qaly']<300])/100
above_30 = np.sum([df["IC_D_50"]/df['total_qaly']<30])/100
print("=========================================================================")
print(f"Using a CEA threshold of $300/DALY averted, there is a {above_300}% probability \nthe intervention is cost effective.")
print("")
print(f"Using a CEA threshold of $30/DALY averted, there is a {above_30}% probability \nthe intervention is cost effective.")
print("A CEA threshold of $30/DALY averted is quite conservative")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(df["total_qaly"], df["IC_D_100"]  ,s=.2, color="b", alpha=.5, label="$100")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values

y3 = df['IC_D_100'].values

ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (mortality benefits included)")
plt.legend(title="CEA Threshold", loc="upper right", markerscale=12)

plt.show();

above_300 = np.sum([df["IC_D_100"]/df['total_qaly']<133])/100
above_30 = np.sum([df["IC_D_100"]/df['total_qaly']<30])/100
print("=========================================================================")
print(f"Using a CEA threshold of $300/DALY averted, there is a {above_300}% probability \nthe intervention is cost effective.")
print("")
print(f"Using a CEA threshold of $30/DALY averted, there is a {above_30}% probability \nthe intervention is cost effective.")
print("A CEA threshold of $30/DALY averted is quite conservative")
print("=========================================================================")

fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(df["total_qaly"], df["IC_D_100"],s=.02, color="g", label="$100")
plt.scatter(df["total_qaly"], df["IC_D_46"] ,s=.02, color="r", label="$46")
plt.scatter(df["total_qaly"], df["IC_D_1"]  ,s=.02, color="b", label="$1")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))


# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100'].values
y2 = df['IC_D_46'].values
y3 = df['IC_D_1'].values

ci_ellipse(x, y1, edgecolor="g")
ci_ellipse(x, y2, edgecolor="r")
ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
plt.xlabel(r'$\Delta$ DALYs')
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (mortality benefits included)")
plt.legend(title="CEA Threshold", loc="upper right", markerscale=12)

plt.show();

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100"],s=.2, color="g", label="$100")
# plt.scatter(df["total_qaly"], df["IC_D_46"] ,s=.02, color="r", label="$46")
sc3 = ax.scatter(df["total_qaly"], df["IC_D_1"]  ,s=.2, color="b", label="$1")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))


# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100'].values
# y2 = df['IC_D_46'].values
y3 = df['IC_D_1'].values

# ci_ellipse(x, y1, edgecolor="g")
# # ci_ellipse(x, y2, edgecolor="r")
# ci_ellipse(x, y3, edgecolor="b")

plt.ylabel(r'$\Delta$ Cost')  # Using the raw string literal r'' and LaTeX syntax for delta
ax.yaxis.set_label_coords(0.25, 0.8)
plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.95, 0.45)
plt.suptitle("CEA Expanded MCCM Madagascar")
plt.title("Societal perspective (mortality benefits included)")
# plt.legend(title="CEA Threshold", loc="upper right", markerscale=12)
leg1 = ax.legend(handles=[sc1, sc3], loc="upper right",
                 title="Compensation",
                 markerscale=12)
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right",
                 title="CE Threshold")

plt.savefig("ceplane_soc.png", bbox_inches="tight")
plt.show();

files.download("ceplane_soc.png")

# $1 compensation for Societal Perspective (No deaths)
budget = list(range(0, 250_000, 25_000))
thresholds = list(range(0,250,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_D_1"] < i).mean()
    proportion_below_threshold = ((df["IC_D_1"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded ICCM Project: With Mortality Benefits")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, 0.990, "Assume compensation of $1 per month")
plt.savefig("ceafc_soc_d_1.png", bbox_inches="tight")

files.download("ceafc_soc_d_1.png")

# $50 compensation for Societal Perspective (No deaths)
budget = list(range(0, 1_250_000, 100_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_D_50"] < i).mean()
    proportion_below_threshold = ((df["IC_D_50"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded MCCM Project: With Mortality Benefits")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, 0.94, "Assume compensation of $50 per month")
plt.savefig("ceafc_soc_d_50.png", bbox_inches="tight")

files.download("ceafc_soc_d_50.png")

# $100 compensation for Societal Perspective (No deaths)
budget = list(range(0, 1_250_000, 100_000))
thresholds = list(range(0,750,1))
data =[]
for i in budget:
  proportions = []
  for threshold in thresholds:
    proportion_below_budget = (df["IC_D_100"] < i).mean()
    proportion_below_threshold = ((df["IC_D_100"]/df["total_qaly"]) < threshold).mean()
    proportion = proportion_below_threshold * proportion_below_budget
    proportions.append(proportion)
  data.append(proportions)
cols = [f"CEAFC_{i}" for i in thresholds]
df3 = pd.DataFrame(data, index=budget, columns=cols)
df3 = df3.T
df3 = df3.add_prefix("CEAFC_")
df3.reset_index(drop=True, inplace=True)
df3[:5]

colnames = df3.columns.tolist()
colnames =["$" + name.split("_")[1] for name in colnames]
df3.plot(figsize=(8,5))
plt.ylabel("Probability cost effective and affordable")
plt.xlabel("CEA Threshold $ per DALY Averted")
plt.grid(alpha=.3)
plt.suptitle("Cost-effectiveness affordability curve: Societal perspective")
plt.title("Madagascar Expanded MCCM Project: With Mortality Benefits")
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
# plt.legend(title="Extra Budget",
#            labels=["$0","$250K", "$500K", "$750K",  "$1 million", ])
plt.legend(title="Extra Budget", labels=colnames, bbox_to_anchor=(1.05,1))
plt.text(0, -0.05, "Assume compensation of $100 per month")
plt.savefig("ceafc_soc_d_100.png", bbox_inches="tight");

# Value of information societal with death
df["nhb_death"] = df["total_qaly"] * 133 - (df["inc_prog"] - df["non_death_savings"] - df["death_costs"])
df["nhb_nondeath"] = df["total_qaly"] * 133 - (df["inc_prog"] - df["non_death_savings"])
voi_d = np.sum([t for t in df["nhb_death"] if t < 0])/10000
voi_nd = np.sum([t for t in df["nhb_nondeath"] if t < 0])/10000
print(f"The value of information from a societal perspective is ${abs(voi_nd):,.2f}")

# Include mortality benefits
# Simulate incremental costs from a societal perspective (no deaths)
etas = [i/10 for i in range(10, 26)]
annual_vsls = []
for eta in etas:
  mc_vsl = us_vsl * (gdp_ppp/us_gdp_pc) ** eta
  mc_vsl_year = mc_vsl/np.sum([(1+discount_rate) ** -t for t in range(rem_life)])
  annual_vsls.append(mc_vsl_year)
annual_vsls

# Include mortality benefits
# Simulate incremental costs from a societal perspective (no deaths)
etas = [i/10 for i in range(10, 26)]

for eta in etas:
  for i in range(1, 151):
    df[f"IC_D_{i}_{eta}"] = df[f"IC_{i}"] - df["non_death_savings"] - df[f'death_costs_{eta}']
df[["IC_ND_100", "IC_100", "IC_D_100_2.5"]][:5]

df[["IC_D_100_1.5", "IC_D_100_2.0", "IC_D_100_2.5"]][:5]

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_1.0"],s=2, color="g", alpha=.5, label="1.0")
sc2 = ax.scatter(df["total_qaly"], df["IC_D_100_1.5"],s=2, color="b", alpha=.5, label="1.5")
sc3 = ax.scatter(df["total_qaly"], df["IC_D_100_2.0"],s=2, color="k", alpha=.5, label="2.0")
sc4 = ax.scatter(df["total_qaly"], df["IC_D_100_2.5"],s=2, color="r", alpha=.5, label="2.5")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel(r'$\Delta$ DALYs')
plt.ylabel(f"$\Delta$ Costs")
plt.suptitle("Madagascar Expanded MCCM Analysis")
plt.title("Impact of different VSL elasticities")

#plt.legend(bbox_to_anchor=(1.2,1.0));
leg1 = ax.legend(handles=[sc1, sc2, sc3, sc4], loc="upper right",
                 title="Elasticity", markerscale=3)
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_1.0'].values
y2 = df['IC_D_100_1.5'].values
y3 = df['IC_D_100_2.0'].values
y4 = df['IC_D_100_2.5'].values

ci_ellipse(x, y1)
ci_ellipse(x, y2)
ci_ellipse(x, y3)
ci_ellipse(x, y4);



result = sum([1 for _, row in df.iterrows() if (row["IC_D_100"] / row["total_qaly"] < 133) and (row["IC_D_100"] < 10000)]) / 100

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_1.0"],s=.2, color="g", alpha=.5, label="1.0")
# sc2 = ax.scatter(df["total_qaly"], df["IC_D_100_1.5"],s= .2, color="r", alpha=.5, label="1.5")
# sc3 = ax.scatter(df["total_qaly"], df["IC_D_100_2.0"],s=2, color="k", alpha=.5, label="2.0")
sc4 = ax.scatter(df["total_qaly"], df["IC_D_100_2.5"],s= .2, color="r", alpha=.5, label="2.5")

l1, = ax.plot([-5_000, 7_500], [-130_000, 195_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 7_500], [-665_000, 997_500], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))

plt.xlabel(r'$\Delta$ DALYs')
ax.xaxis.set_label_coords(0.9, 0.55)
plt.ylabel(f"$\Delta$ Costs")
ax.yaxis.set_label_coords(0.3, 0.7)

plt.suptitle("Madagascar Expanded MCCM Analysis")
plt.title("Impact of different VSL elasticities")

#plt.legend(bbox_to_anchor=(1.2,1.0));
leg1 = ax.legend(handles=[sc1, sc4], loc="upper right",
                 title="Elasticity", markerscale=12)
ax.add_artist(leg1)
leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_1.0'].values
# y2 = df['IC_D_100_1.5'].values
# y3 = df['IC_D_100_2.0'].values
y4 = df['IC_D_100_2.5'].values

# ci_ellipse(x, y1)
# ci_ellipse(x, y2)
# ci_ellipse(x, y3)
# ci_ellipse(x, y4);
plt.savefig("elasticity_soc.png", bbox_inches="tight")

files.download("elasticity_soc.png")

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_1.0"],s=2, color="g", alpha=.5)

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel(r'$\Delta$ DALYs')
plt.ylabel(f"$\Delta$ Costs")
plt.suptitle("Madagascar MCCM CEA")
plt.title("Elasticity of 1")

plt.legend(bbox_to_anchor=(1.0,1.0), title="CE Threshold");
# leg1 = ax.legend(handles=[sc1, sc2, sc3, sc4], loc="upper right", title="Elasticity")
# ax.add_artist(leg1)
# leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_1.0'].values

ci_ellipse(x, y1);

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_1.5"],s=2, color="g", alpha=.5)

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel(r'$\Delta$ DALYs')
plt.ylabel(f"$\Delta$ Costs")
plt.suptitle("Madagascar MCCM CEA")
plt.title("Elasticity of 1.5")

plt.legend(bbox_to_anchor=(1.0,1.0), title="CE Threshold");
# leg1 = ax.legend(handles=[sc1, sc2, sc3, sc4], loc="upper right", title="Elasticity")
# ax.add_artist(leg1)
# leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_1.5'].values

ci_ellipse(x, y1);

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_2.0"],s=2, color="g", alpha=.5)

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel(r'$\Delta$ DALYs')
plt.ylabel(f"$\Delta$ Costs")
plt.suptitle("Madagascar MCCM CEA")
plt.title("Elasticity of 2.0")

plt.legend(bbox_to_anchor=(1.0,1.0), title="CE Threshold");
# leg1 = ax.legend(handles=[sc1, sc2, sc3, sc4], loc="upper right", title="Elasticity")
# ax.add_artist(leg1)
# leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_2.0'].values

ci_ellipse(x, y1);

fig, ax = plt.subplots(figsize=(8,5))
sc1 = ax.scatter(df["total_qaly"], df["IC_D_100_2.5"],s=2, color="g", alpha=.5)

l1, = ax.plot([-5_000, 12_000], [-130_000, 312_000], label="$26/DALY", ls="--")
l2, = ax.plot([-5_000, 10_000], [-665_000, 1_330_000], label="$133/DALY", ls="--")
l3, = ax.plot([-5_000, 6_000], [-3_450_000, 4_140_000], label="$690/DALY", ls="--")

ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))
for sp in ["top", "right"]:
  ax.spines[sp].set_visible(False)
plt.grid(alpha=.2)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.xlabel(r'$\Delta$ DALYs')
plt.ylabel(f"$\Delta$ Costs")
plt.suptitle("Madagascar MCCM CEA")
plt.title("Elasticity of 2.5")

plt.legend(bbox_to_anchor=(1.0,1.0), title="CE Threshold");
# leg1 = ax.legend(handles=[sc1, sc2, sc3, sc4], loc="upper right", title="Elasticity")
# ax.add_artist(leg1)
# leg2 = ax.legend(handles=[l1, l2, l3], loc ="lower right", title="CE Threshold")
# Convert DataFrame columns to NumPy arrays
x = df['total_qaly'].values
y1 = df['IC_D_100_2.5'].values

ci_ellipse(x, y1);

df[["death_costs_1.0", "death_costs_1.5","death_costs_2.0","death_costs_2.5",]][:10]

df["inc_prog_death"] = df["inc_prog"] - df["non_death_savings"] - df["death_costs_1.4"]
df['inc_prog_no_death'] = df["inc_prog"] - df["non_death_savings"]

df["admissions_averted"].mean()/.65


# ICERs
mean_icer = df["inc_prog_no_death"].mean()/df["total_qaly"].mean()
min_icer = df["inc_prog_no_death"].min()/df["total_qaly"].mean()
max_icer = df["inc_prog_no_death"].max()/df["total_qaly"].mean()
print(f"Societal perspective no death: {mean_icer:,.2f} [{min_icer:,.2f}, {max_icer:,.2f}]")

# ICERs
mean_icer = df["inc_prog_death"].mean()/df["total_qaly"].mean()
min_icer = df["inc_prog_death"].min()/df["total_qaly"].mean()
max_icer = df["inc_prog_death"].max()/df["total_qaly"].mean()
print(f"Societal perspective: {mean_icer:,.2f} [{min_icer:,.2f}, {max_icer:,.2f}]")

def univariate_sensitivity(df, univ_df, col1, col2, col3):
    """
    Calculate the maximum and minimum incremental cost-effectiveness ratios (ICERs)
    for two different columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The main DataFrame containing the orginal data.
        univ_df (pandas.DataFrame): The DataFrame containing the column values for minimum and maximum.
        column_name1 (str): The name of the first column to use for the minimum and maximum values.
        column_name2 (str): The name of the second column to use for the minimum and maximum values.

    Returns:
        tuple: A tuple containing four values:
            - icer_max1 (float): The maximum ICER for the first column.
            - icer_min1 (float): The minimum ICER for the first column.
            - icer_max2 (float): The maximum ICER for the second column.
            - icer_min2 (float): The minimum ICER for the second column.
    """
    min_val = univ_df[col1].iloc[1]
    max_val = univ_df[col1].iloc[2]

    max_x = df[col2] * max_val
    min_x = df[col2] * min_val

    max_y = df['inc_prog_no_death'] + df[col3] - max_x
    min_y = df['inc_prog_no_death'] + df[col3] - min_x

    icer_max = np.mean(max_y) / df['total_qaly'].mean()
    icer_min = np.mean(min_y) / df['total_qaly'].mean()

    return icer_max, icer_min

# Create an empty dictionary that we will add minimum and maximum values to
univ_sens_dict = {}

#Admission costs
result = univariate_sensitivity(df, univ_df, "admission_cost", "admissions_averted", "admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Costs"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admission costs: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#Admissions averted
result = univariate_sensitivity(df, univ_df,  "admissions_averted", "admission_cost","admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Averted"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admissions averted: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#QALYs
univ_qaly_min = df['inc_prog_no_death'].mean()/df['total_qaly'].quantile(.975)
univ_qaly_max = df['inc_prog_no_death'].mean()/df['total_qaly'].quantile(.025)
univ_sens_dict["Utility"] = {"min": univ_qaly_max, "max":univ_qaly_min}
print(f"One way sensitivity analyses varying QALY: $[{univ_qaly_max:,.2f}, {univ_qaly_min:,.2f}]")

min_val = univ_df["change_opd"].iloc[1]
max_val = univ_df["change_opd"].iloc[2]
min_x = min_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_x = max_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_y = df['inc_prog_no_death'] + df['opd_costs_averted'] - min_x
min_y = df['inc_prog_no_death'] + df['opd_costs_averted'] - max_x
univ_opd_averted_max = np.mean(max_y)/df['total_qaly'].mean()
univ_opd_averted_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_opd_averted_max)
print(univ_opd_averted_min)
univ_sens_dict["OPD Costs Averted"] = {"min": univ_opd_averted_min, "max":univ_opd_averted_max}

# Inpatient consumables
mean_val = df['admission_cost'] - df['ipd_cons_lab']
min_val = mean_val + univ_df['ipd_cons_lab'][1]
max_val = mean_val + univ_df['ipd_cons_lab'][2]
min_y = df["inc_prog_no_death"] + df["admissions_costs_averted"] - (df['admissions_averted'] * max_val)
max_y = df["inc_prog_no_death"] + df["admissions_costs_averted"] - (df['admissions_averted'] * min_val)
univ_consum_max = np.mean(max_y)/df['total_qaly'].mean()
univ_consum_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_consum_min, univ_consum_max)

univ_sens_dict["IP Consumables"] = {"min": univ_consum_min, "max":univ_consum_max}
max_prod = (df['inc_prog_no_death'].mean() + df["non_death_savings"].mean() - df["non_death_savings"].min())/df["total_qaly"].mean()
min_prod = (df['inc_prog_no_death'].mean() + df["non_death_savings"].mean() - df["non_death_savings"].max())/df["total_qaly"].mean()
print(min_prod, max_prod)
univ_sens_dict["Productivity"] = {"min": min_prod, "max":max_prod}


# Calculate mean value
mean_value = df["inc_prog_no_death"].mean()/df["total_qaly"].mean()
# Calculate the range for each bar and sort the dictionary based on the range
univ_sens_dict = dict(sorted(univ_sens_dict.items(), key=lambda item: item[1]['max'] - item[1]['min'], reverse=False))

# Extract labels, minimum, and maximum values from the dictionary
labels = list(univ_sens_dict.keys())
min_values = [data['min'] for data in univ_sens_dict.values()]
max_values = [data['max'] for data in univ_sens_dict.values()]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot horizontal bars
bar1 = ax.barh(labels, [mean_value - min_val for min_val in min_values], left=min_values,
        color='lightcoral', label='Minimum', height=0.4)
bar2 = ax.barh(labels, [max_val - mean_value for max_val in max_values], left=mean_value,
        color='lightblue', label='Maximum', height=0.4)

# Add mean line
ax.axvline(x=mean_value, color='gray', linestyle='--', label='Mean')

# Add labels, legend, and title
ax.set_xlabel('ICER')
ax.set_ylabel('')
ax.set_title('Tornado Diagram: Societal Perspective without Mortality')
ax.legend()
for rect, min_val, max_val in zip(bar1.patches, min_values, max_values):
    width = rect.get_width()
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${min_val:,.0f}', ha='right', va='center', color='black')

for rect, min_val, max_val in zip(bar2.patches, min_values, max_values):
    width = rect.get_width()
    if max_val < 0:
      color = "red"
    else:
      color = "black"
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${max_val:,.0f}', ha='left', va='center', color=color)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(lw=.2, alpha=.2)
# Show plot
plt.savefig("tornado_soc_no_death.png", bbox_inches="tight")
plt.show()

files.download("tornado_soc_no_death.png")

def univariate_sensitivity(df, univ_df, col1, col2, col3):
    """
    Calculate the maximum and minimum incremental cost-effectiveness ratios (ICERs)
    for two different columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The main DataFrame containing the orginal data.
        univ_df (pandas.DataFrame): The DataFrame containing the column values for minimum and maximum.
        column_name1 (str): The name of the first column to use for the minimum and maximum values.
        column_name2 (str): The name of the second column to use for the minimum and maximum values.

    Returns:
        tuple: A tuple containing four values:
            - icer_max1 (float): The maximum ICER for the first column.
            - icer_min1 (float): The minimum ICER for the first column.
            - icer_max2 (float): The maximum ICER for the second column.
            - icer_min2 (float): The minimum ICER for the second column.
    """
    min_val = univ_df[col1].iloc[1]
    max_val = univ_df[col1].iloc[2]

    max_x = df[col2] * max_val
    min_x = df[col2] * min_val

    max_y = df['inc_prog_death'] + df[col3] - max_x
    min_y = df['inc_prog_death'] + df[col3] - min_x

    icer_max = np.mean(max_y) / df['total_qaly'].mean()
    icer_min = np.mean(min_y) / df['total_qaly'].mean()

    return icer_max, icer_min

# Create an empty dictionary that we will add minimum and maximum values to
univ_sens_dict = {}


#Admission costs
result = univariate_sensitivity(df, univ_df, "admission_cost", "admissions_averted", "admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Costs"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admission costs: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#Admissions averted
result = univariate_sensitivity(df, univ_df,  "admissions_averted", "admission_cost","admissions_costs_averted")
univ_adm_cost_max = result[1]
univ_adm_cost_min = result[0]
univ_sens_dict["Admission Averted"] = {"min": result[0], "max":result[1]}
print(f"One-way sensitivity analysis - admissions averted: $[{result[0]:,.2f}, {result[1]:,.2f}]")

#QALYs
univ_qaly_min = df['inc_prog_death'].mean()/df['total_qaly'].quantile(.975)
univ_qaly_max = df['inc_prog_death'].mean()/df['total_qaly'].quantile(.025)
univ_sens_dict["Utility"] = {"min": univ_qaly_max, "max":univ_qaly_min}
print(f"One way sensitivity analyses varying QALY: $[{univ_qaly_max:,.2f}, {univ_qaly_min:,.2f}]")

min_val = univ_df["change_opd"].iloc[1]
max_val = univ_df["change_opd"].iloc[2]
min_x = min_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_x = max_val * (opd_costs_2022 + total_rdt + pcm + art_lume +  0.3 *(hemogram + microscopy)) * 1.05
max_y = df['inc_prog_death'] + df['opd_costs_averted'] - min_x
min_y = df['inc_prog_death'] + df['opd_costs_averted'] - max_x
univ_opd_averted_max = np.mean(max_y)/df['total_qaly'].mean()
univ_opd_averted_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_opd_averted_max)
print(univ_opd_averted_min)
univ_sens_dict["OPD Costs Averted"] = {"min": univ_opd_averted_min, "max":univ_opd_averted_max}

# Inpatient consumables
mean_val = df['admission_cost'] - df['ipd_cons_lab']
min_val = mean_val + univ_df['ipd_cons_lab'][1]
max_val = mean_val + univ_df['ipd_cons_lab'][2]
min_y = df["inc_prog_death"] + df["admissions_costs_averted"] - (df['admissions_averted'] * max_val)
max_y = df["inc_prog_death"] + df["admissions_costs_averted"] - (df['admissions_averted'] * min_val)
univ_consum_max = np.mean(max_y)/df['total_qaly'].mean()
univ_consum_min = np.mean(min_y)/df['total_qaly'].mean()
print(univ_consum_min, univ_consum_max)
univ_sens_dict["IP Consumables"] = {"min": univ_consum_min, "max":univ_consum_max}

max_death = (df['inc_prog_death'].mean() + df["death_costs_1.4"].mean() - df["death_costs_2.5"].mean())/df["total_qaly"].mean()
min_death = (df['inc_prog_death'].mean() + df["death_costs_1.4"].mean() - df["death_costs_1.0"].mean())/df["total_qaly"].mean()
print(max_death, min_death)
univ_sens_dict["Elasticity"] = {"min": min_death, "max":max_death}

max_prod = (df['inc_prog_death'].mean() + df["non_death_savings"].mean() - df["non_death_savings"].min())/df["total_qaly"].mean()
min_prod = (df['inc_prog_death'].mean() + df["non_death_savings"].mean() - df["non_death_savings"].max())/df["total_qaly"].mean()
print(min_prod, max_prod)
univ_sens_dict["Productivity"] = {"min": min_prod, "max":max_prod}


# Calculate mean value
mean_value = df["inc_prog_death"].mean()/df["total_qaly"].mean()
# Calculate the range for each bar and sort the dictionary based on the range
univ_sens_dict = dict(sorted(univ_sens_dict.items(), key=lambda item: item[1]['max'] - item[1]['min'], reverse=False))

# Extract labels, minimum, and maximum values from the dictionary
labels = list(univ_sens_dict.keys())
min_values = [data['min'] for data in univ_sens_dict.values()]
max_values = [data['max'] for data in univ_sens_dict.values()]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot horizontal bars
bar1 = ax.barh(labels, [mean_value - min_val for min_val in min_values], left=min_values,
        color='lightcoral', label='Minimum', height=0.4)
bar2 = ax.barh(labels, [max_val - mean_value for max_val in max_values], left=mean_value,
        color='lightblue', label='Maximum', height=0.4)

# Add mean line
ax.axvline(x=mean_value, color='gray', linestyle='--', label='Mean')

# Add labels, legend, and title
ax.set_xlabel('ICER')
ax.set_ylabel('')
ax.set_title('Tornado Diagram: Societal Perspective with Mortality')
ax.legend()
for rect, min_val, max_val in zip(bar1.patches, min_values, max_values):
    width = rect.get_width()
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${min_val:,.0f}', ha='right', va='center', color='black')

for rect, min_val, max_val in zip(bar2.patches, min_values, max_values):
    width = rect.get_width()
    if max_val < 0:
      color = "red"
    else:
      color = "black"
    ax.text(rect.get_x() + width, rect.get_y() + rect.get_height() / 2,
            f'${max_val:,.0f}', ha='left', va='center', color=color)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
plt.grid(lw=.2, alpha=.2)
# Show plot
plt.savefig("tornado_soc_death.png", bbox_inches="tight")
plt.show()

files.download("tornado_soc_death.png")

from google.colab import drive
drive.mount('/content/gdrive')

cd "/content/gdrive/MyDrive/Colab Notebooks"

!jupyter nbconvert --to html "Copy of mada_mccm.ipynb"


