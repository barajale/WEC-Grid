%% LUPA
%% Simulation Data
simu = simulationClass();        
simu.simMechanicsFile = 'W2G_ss_lupa.slx';%Location of Simulink Model File with PTO-Sim  
simu.explorer = 'off';                  % Turn SimMechanics Explorer (on/off)
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 0;                      % Wave Ramp Time [s]
simu.endTime = simLength;                     % Simulation End Time [s]
simu.solver = 'ode4';    % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = dt;
simu.cicEndTime = 10; 	

%% Wave Information
waves = waveClass(waveClassType);
waves.height = waveHeight;
waves.period = wavePeriod;
waves.spectrumType = spectrumType;
waves.phaseSeed = waveSeed;

% %% Body Data
% %% Body 1: Float
% body(1) = bodyClass('./hydroData/floatspar_D1_14m_d2_78m.h5');
% body(1).geometryFile = './geometry/LUPA_Fall2022_float_geometry.stl';
% body(1).mass = 'equilibrium';
% body(1).viz.color = [255/256 127/256 36/256];
% body(1).inertia = [64.60 65.06 17.83];                      % [kg-m^2] As measured from dry swing tests
% body(1).quadDrag.cd = [0.54 0.54 0.15 0.54 0.54 0.15];          % [-] Quadratic drag coefficient Cd as found from Gu et al 2018
% body(1).quadDrag.area = [0.368 0.368 0.785 0.368 0.368 0.785];  % [m^2] Characteristic area in relevant plane

% %% Body 2: Spar
% body(2) = bodyClass('./hydroData/floatspar_D1_14m_d2_78m.h5');
% body(2).geometryFile = './geometry/LUPA_spar_D1_14m_full.stl';
% body(2).mass = 202.21;                                             % [kg] Positively bouyant spar.
% body(2).viz.color = [211/256 211/256 211/256];
% body(2).inertia = [304.786 305.250 15.775];                      % [kg-m^2] As measured from dry swing tests
% body(2).quadDrag.cd = [0.6 0.6 2.8 0.6 0.6 2.8];                    % [-] Quadratic drag coefficient Cd as found from Beatty 2015 and Singh & Mittal 2005
% body(2).quadDrag.area = [0.558 0.558 0.636 0.558 0.558 0.636];      % [m^2] Characteristic area in relevant plane
% body(2).setInitDisp([0 0 0],[0 0 0 0],[0 0 -0.22]);                 % [m] Initial Displacement  Set to engage mooring lines for pre-tension.



%% Body Data
%% Body 1: Float
body(1) = bodyClass('.\hydroData\floatspar_20m.h5');
body(1).geometryFile = '.\geometry\LUPA_float_20m.stl';
body(1).mass = 'equilibrium';
body(1).viz.color = [255/256 127/256 36/256];
body(1).inertia = [206720000 208192000 57056000];       % [kg-m^2] scaled 20x from Lab-scale measured dry swing tests
body(1).quadDrag.cd = [0.54 0.54 0.15 0.54 0.54 0.15];  % [-] Quadratic drag coefficient Cd as found from Gu et al 2018
body(1).quadDrag.area = [73 73 314.16 73 73 314.16];    % [m^2] Characteristic area in relevant plane 

%% Body 2: Spar
body(2) = bodyClass('.\hydroData\floatspar_20m.h5');
body(2).geometryFile = '.\geometry\LUPA_spar_20m_full.stl';
body(2).mass = 1617680;   %'equilibrium';                        % [kg] Positively bouyant spar. Scaled 20x
body(2).viz.color = [211/256 211/256 211/256];
body(2).inertia = [975315200 976800000 50480000];                % [kg-m^2] scaled 20x from Lab-scale measured dry swing tests
body(2).quadDrag.cd = [0.6 0.6 2.8 0.6 0.6 2.8];                 % [-] Quadratic drag coefficient Cd as found from Beatty 2015 and Singh & Mittal 2005
body(2).quadDrag.area = [103.78 103.78 130 103.78 103.78 130];   % [m^2] Characteristic area in relevant plane
body(2).setInitDisp([0 0 0],[0 0 0 0],[0 0 -0.6]);               % [m] Initial Displacement  Set to engage mooring lines for pre-tension.


%% PTO and Constraint Parameters
% Translational Constraint
constraint(1) = constraintClass('Constraint1'); 
constraint(1).location = [0 0 0]; 

% Translational PTO
pto(1) = ptoClass('PTO1');           	% Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                           % PTO Damping [N/(m/s)]
pto(1).location = [0 0 0];                   % PTO Location [m]





%% Mooring Matrix

R = 6.5;                                   % [m] Radius of mooring plate scaled 10x
mooring(1) = mooringClass('mooring');       % Initialize mooringClass

%%This is the combined equivalent stiffness of all 4 springs as calculated
%%from their equilbrium location and angle. Scaled up 20x from lab-scale
mooring(1).matrix.stiffness(1,1) = 1145600;   % [N/m]
mooring(1).matrix.stiffness(2,2) = 1145600;   % [N/m]
mooring(1).matrix.stiffness(3,3) = 224800;   % [N/m] 
mooring(1).matrix.stiffness(4,4) = 32712000*R;   % [N/deg] Assumming small angle approximations acting axially
mooring(1).matrix.stiffness(5,5) = 32712000*R;   % [N/deg] Assumming small angle approximations acting axially
mooring(1).matrix.stiffness(6,6) = 32712000*R;   % [N/deg] Assumming small angle approximations acting axially
mooring(1).location = [0 0 -14.4-0.6];      % [m] Distance in meters from the still water line down to the mooring connection point when the spar is initially displaced.




%%  PTO-Sim Setup: 
%% Linear Generator PTO-Sim  
 
ptoSim(1) = ptoSimClass('PTOSim');
ptoSim(1).number  = 1;
ptoSim(1).type = 9; %Direct drive linear generator


%% Linear Generator

%params from doi: 10.1109/ECCE.2009.5316224.
%Small-scale generator parameters appropriate for LUPA device size
ptoSim(1).directLinearGenerator.Bfric = 0;%100;         % Friction coefficient
ptoSim(1).directLinearGenerator.tau_p = 0.072;          % Magnet pole pitch [m]
ptoSim(1).directLinearGenerator.lambda_fd = 8;          % Flux linkage of the stator d winding due to flux produced by the rotor magnets [Wb-turns]
ptoSim(1).directLinearGenerator.lambda_sq_0 = 0;
ptoSim(1).directLinearGenerator.lambda_sd_0 = ptoSim.directLinearGenerator.lambda_fd;  % (recognizing that the d-axis is always aligned with the rotor magnetic axis                        
ptoSim(1).directLinearGenerator.Rs = 4.58;              % Winding resistance [ohm]
ptoSim(1).directLinearGenerator.Ls = 0.285;             % Inductance of the coil [H], per-phase inductance *3/2
ptoSim(1).directLinearGenerator.theta_d_0 = 0;



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


%% Onboard energy storage
%energy storage system
ess.Vdc_0 = gsc.Vmag*sqrt(2)*1.25; %V, 25% higher than grid voltage
ess.C = 88; %F
ess.Vdc_del = ess.Vdc_0-gsc.Vmag*sqrt(2); %max deviation from nominal voltage. When determining this value, ensure (Vdc_nom - Vdc_del) > gsc.Vmag*sqrt(2) ?






