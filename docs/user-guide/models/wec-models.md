# WEC Models

WEC-Grid includes validated wave energy converter models from academic research, providing realistic device characteristics for integration studies.

## Available Models

### RM3 Reference Model
- **Description**: Two-body point absorber developed by Sandia National Laboratories
- **Type**: Point absorber with vertical motion
- **Power Rating**: 1 MW nominal
- **Applications**: Offshore deployment scenarios
- **Validation**: Extensively validated against experimental data
- **Citation**: [RM3 reference model citation]

**Key Characteristics:**
- Dual-body design with float and submerged reaction body  
- Power take-off through relative motion
- Representative of commercial point absorber designs
- Well-documented hydrodynamic coefficients

### LUPA Model
- **Description**: [Add description of LUPA model]
- **Type**: [Add type information]
- **Power Rating**: [Add power rating]
- **Applications**: [Add application scenarios]
- **Citation**: https://github.com/PMEC-OSU/LUPA_WEC-Sim/tree/main

using the Two body Heave only 14m verison found here

https://github.com/PMEC-OSU/LUPA_WEC-Sim/tree/main/MULTIPHYSICSLUPA3%20Spring%202024

should mention that this model is producing about 100 w of power, 

add screenshot of LUPA simulation results in Watts over time 

for examples the LUPA was scaled up. 


Citing LUPA
Publication [1] B. Bosma, C. Beringer, M. Leary, B. Robertson. “Design and modeling of a laboratory scale WEC point absorber” in Proceedings of the 14th European Wave and Tidal Energy Conference, EWTEC 2021, Plymouth, UK, 2021.

LUPA v1.0 [1] Bret Bosma. (2022, April), LUPA (Version v1.0), DOI

## Model Integration

WEC models in WEC-Grid provide:

- **Hydrodynamic modeling**: Integration with WEC-Sim for device-level physics
- **Power conversion**: Realistic power take-off system modeling
- **Grid interface**: Appropriate electrical characteristics for grid connection
- **Scalability**: Support for single devices and arrays

## Custom Modifications

The standard academic models have been enhanced for grid integration:

- **Electrical interface modeling**: Added grid-connection components
- **Control system integration**: Incorporated grid-friendly control strategies
- **Array modeling**: Support for multiple device deployments
- **Environmental coupling**: Integration with wave resource data


need to talk about the custom PTOsim stuff 


for lupa

%% Back-to-back converter parameters

%WEC-side converter - small scale appropriate for LUPA
wsc.Bdamp = 1e3;%97e3; %Resistive damping coef - 1 kW scale
wsc.Kdamp = 0; %Reactive damping coef
wsc.Fpto_lim = pi*ptoSim(1).directLinearGenerator.lambda_fd^2/ptoSim(1).directLinearGenerator.Ls/ptoSim(1).directLinearGenerator.tau_p/2*0.999;
%NOTE: the Fpto limit is reduced to 99.9% to avoid a singularity in the
%simulation. Theoretically this shouldn't be needed but I think it happens
%due to small error accumulation from numerical calculations

%grid-side converter - small scale
gsc.Prated = 1e3; % 1 kW rated power for small LUPA device
gsc.Vmag = 480*1.1; %V, rms, l-l, 10% higher voltage than grid Vnom
gsc.Ilim = gsc.Prated/gsc.Vmag; %A, rms
gsc.Tavg = 5*60; %averaging period, s

%voltage correction PI controller
gsc.kp = gsc.Prated;
gsc.ki = 0;

rm3 

%% Back-to-back converter parameters

%WEC-side converter
wsc.Bdamp = 100e3;%97e3; %Resistive damping coef
wsc.Kdamp = 0; %Reactive damping coef
wsc.Fpto_lim = pi*ptoSim(1).directLinearGenerator.lambda_fd^2/ptoSim(1).directLinearGenerator.Ls/ptoSim(1).directLinearGenerator.tau_p/2*0.999;
%NOTE: the Fpto limit is reduced to 99.9% to avoid a singularity in the
%simulation. Theoretically this shouldn't be needed but I think it happens
%due to small error accumulation from numerical calculations

%grid-side converter
gsc.Prated = 60e3;
gsc.Vmag = 480*1.1; %V, rms, l-l, 10% higher voltage than grid Vnom
gsc.Ilim = gsc.Prated/gsc.Vmag; %A, rms
gsc.Tavg = 5*60; %averaging period, s

%voltage correction PI controller
gsc.kp = gsc.Prated;
gsc.ki = 0;



add screenshot of M2G simulink stuff





## Usage

WEC models are selected when configuring WEC devices:



## Validation

All WEC models included in WEC-Grid have been validated against:

- Original experimental or numerical data
- Independent modeling results
- Field deployment data (where available)

## References

[Include proper citations for each WEC model and any custom modifications]
