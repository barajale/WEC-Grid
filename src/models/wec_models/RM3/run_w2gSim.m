clear, close all, format compact

%w2g inputs
sim_id = 1;
simLength = 1*3600; %[s]
Tsample = 5*60; %[s]
waveHeight = 2.5; %[m]
wavePeriod = 8; %[s]
waveSeed = 40; %integer
model = "RM3";


%run
% [m2g_out] = w2gSim(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);
[m2g_out] = w2gSim(sim_id,simLength,Tsample,waveHeight,wavePeriod, waveSeed, model);