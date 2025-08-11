function [m2g_out] = w2gSim(sim_id,simLength,Tsample,waveHeight,wavePeriod,waveSeed, model)
% runs all the code for generating grid power from a WEC
% sim_id: 
% simLength: the duration to run the sim [s]
% Tsample: the sampling period of the data to be sent to the power flow
% solver [s]
% waveHeight: wave height [m]
% wavePeriod: wave period [s]
% waveSeed (optional): initial seed used to generate the wave profile [integer]

%if no seed given, generate one
if exist('waveSeed') == 0
    rng()
    waveSeed = randi(1e9);
else
end

%make sure variables are the right type
simLength = double(simLength);
Tsample = double(Tsample);
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

%downsample data to desired resolution for PSSE
m2g_out.Pgrid_ds = DownSampleTS(m2g_out.Pgrid,Tsample,1);
m2g_out.Qgrid_lim_ds = DownSampleTS(m2g_out.Qgrid_lim,Tsample,1);

%add wec-id to struct
m2g_out.sim_id = sim_id;

m2g_out.model = model; %add model name to output

% Raw wave elevation η(t) from WEC-Sim
m2g_out.Hs        = waves.height;     % m
m2g_out.Tp        = waves.period;     % s
m2g_out.seed      = waves.phaseSeed;  % int
m2g_out.t_eta = waves.waveAmpTime(:,1);
m2g_out.eta   = waves.waveAmpTime(:,2);
 

%% Plots

%plot_W2G_waveforms

end



