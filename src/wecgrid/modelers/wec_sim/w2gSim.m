function [m2g_out] = w2gSim(simLength,dt,spectrumType,waveClassType,waveHeight,wavePeriod,waveSeed)


% Convert inputs to appropriate types
simLength = double(simLength);
dt = double(dt);
spectrumType = char(spectrumType);
waveClassType = char(waveClassType);
waveHeight = double(waveHeight);
wavePeriod = double(wavePeriod);
waveSeed = int32(waveSeed);



%% Initialization of WEC-Sim

% Initialize WEC-Sim
run('wecSimInputFile.m');


%makes 'initializeWecSim' call 'wecSimInputFile' instead of other options
runWecSimCML = 1;

%initialize
run('initializeWecSim');

% Clear variables after initialization (but keep simu for later use)
clear waves body cable pto constraint ptoSim mooring


%% Run W2G sim
sim(simu.simMechanicsFile, [], simset('SrcWorkspace', 'current'));

%% Post-Processing

enableUserDefinedFunctions = 0; %set whether the UDFs are called, 0 to not call

simu.caseDir = working_dir; %set case directory to working directory
%run post-sim script
run('stopWecSim');


%add simulation parameters to struct for database storage
%m2g_out.model = 'RM3'; %hardcoded model name for this device directory (char array)
m2g_out.simLength = simLength; %add simulation duration
m2g_out.dt = dt; %add sampling period

% Wave parameters already included:
m2g_out.Hs        = waves.height;     % m
m2g_out.Tp        = waves.period;     % s
m2g_out.seed      = waves.phaseSeed;  % int
m2g_out.spectrumType = spectrumType; % char array
m2g_out.waveClass = waveClassType; % char array
m2g_out.t_eta = waves.waveAmpTime(:,1);
m2g_out.eta   = waves.waveAmpTime(:,2);
 

%% Plots

%plot_W2G_waveforms

end



