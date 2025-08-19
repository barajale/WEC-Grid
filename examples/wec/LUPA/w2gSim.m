function [m2g_out] = w2gSim(simLength,dt,spectrumType,waveClassType,waveHeight,wavePeriod,waveSeed)

simLength = double(simLength);
dt = double(dt);
spectrumType = char(spectrumType);
waveClassType = char(waveClassType);
waveHeight = double(waveHeight);
wavePeriod = double(wavePeriod);
waveSeed = int32(waveSeed);


%% Initialization of WEC-Sim

% Initialize WEC-Sim
run('wecSimInputFile');
clear simu waves body cable pto constraint ptoSim mooring 

%makes 'initializeWecSim' call 'wecSimInputFile' instead of other options
runWecSimCML = 1;

%initialize
run('initializeWecSim');

%% Run W2G sim
sim(simu.simMechanicsFile, [], simset('SrcWorkspace', 'current'));

%% Post-Processing

enableUserDefinedFunctions = 0; %set whether the UDFs are called, 0 to not call

%run post-sim script
run('stopWecSim');


%add simulation parameters to struct for database storage
m2g_out.model = 'LUPA'; %hardcoded model name for this device directory (char array)
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
 

end


