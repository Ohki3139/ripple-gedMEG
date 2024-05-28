%% Ripple-gedMEG
%
% This simulation file ("ripple_gedMEG.m") calculates multivariate sharp-wae ripples via generalized eigendecomposition.
% This paper is currently under review as Ohki et al. "Multivariate sharp-wave ripples in schizophrenia during awake state"
% 
% Questions -> takefumi2ohki@gmail.com
%
% Note:  This code is based and modified on Michael X Cohen's excellent tutorial paper  on generalized eigenvalue decomposition. 
%           Please also refer to this paper:
%           "A tutorial on generalized eigendecomposition for denoising, 
%           contrast enhancement, and dimension reduction in multichannel
%           electrophysiology.", NeuroImage (2022)


%% Setup
clear all, close all

% load mat file containing MEG, leadfield and channel locations
load emptyMEG

% Basics parameters 4 simulations
MEG.srate   = 1000; % Hz
MEG.trials   =  50;
MEG.pnts    = 1000;
MEG.times  =  ( 0 : MEG.pnts -1 ) / MEG.srate;

% Index 4 dipole location associated with ripples
% Although we set the dipole location based on the empirical findings,
% you can try different dipoles positions.

dipoleLoc = 1560; % right MTL including hippocampus
nDipoles   = 1;       % number of distinct dipole locations

% normalize dipoles (normal to the surface of the head)
lf.GainN = bsxfun(@times, squeeze( lf.Gain(:,1,:)), lf.GridOrient(:,1)') + bsxfun(@times,squeeze( lf.Gain(:,2,:)),lf.GridOrient(:,2)') + bsxfun(@times,squeeze( lf.Gain(:,3,:) ), lf.GridOrient(:,3)' );


%% Generate random dipole data in the brain and target signal.
% create 1000 time points of random dipole data in the brain

% noise level
scaler4noise = 0.9;
dipole_data = scaler4noise*randn( length(lf.Gain), 1000);

% specify the frequency (Hz) 4 the target event
frex = 120;

% duration, starting and end time  4 the target event
duration = 50; 
st_time   = 501; ed_time  = st_time + duration;

% Create a simulated ripple with some noises in the specified dipole location
% and add it to dipole data as Ground-truth data 

% Simulation 1: signal only consisting of ripple with some noise
% NOTE: the use of the randn function may lead to slight variations in the results as it generates random numbers. 
% Therefore, the execution outcomes may vary slightly depending on the specific run.
dipole_data(dipoleLoc, st_time : ed_time ) = 15*sin(2*pi*frex*( 0 : duration ) /MEG.srate ) + scaler4noise*randn(1,  duration + 1) ;

% Simulation 2: signal consisting of phase amplitude coupling (e.g., spindle and ripple)
% parameter 4 the coupling strength
% nonmodulatedamplitude = 5; % increase this to get less modulation; you'll see that this is reflected in the MI value
% 
% % lower frex (i.e., nesting phase frex) and higher frex (i.e., nested amplitude frex) 
% Phase_Modulating_Freq = 15;
% Amp_Modulated_Freq    = frex;

% creation simulation data
% dipole_data(dipoleLoc, st_time : ed_time ) =  (10*(sin(2*pi*( 0 : duration )*Phase_Modulating_Freq/MEG.srate)+1) ... 
%                                                                          + nonmodulatedamplitude*0.1).*sin(2*pi*( 0 : duration )*Amp_Modulated_Freq/MEG.srate)...
%                                                                          + sin(2*pi*( 0 : duration )*Phase_Modulating_Freq/MEG.srate) ...
%                                                                          + scaler4noise*randn(1, length(st_time: ed_time) );

% project dipole data to MEG data
MEG.data = lf.GainN*dipole_data;

% time index based on sampling rate 
MEG.times = ( 0 : size(MEG.data,2) -1 ) / MEG.srate;

% plot brain dipoles
figure(1), clf, 

subplot(221)
plot3( lf.GridLoc(:, 1), lf.GridLoc(:, 2), lf.GridLoc(:, 3), 'o')
hold on
plot3( lf.GridLoc(:,1), lf.GridLoc(:, 2), lf.GridLoc(:, 3), 'ko'), hold on
plot3( lf.GridLoc(dipoleLoc, 1), lf.GridLoc(dipoleLoc, 2), lf.GridLoc(dipoleLoc, 3),'ro','markerfacecolor','r','markersize',10)
rotate3d on, axis square, axis off
title('Dipole locations in the brain')

% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(222)
topoplotIndie( lf.GainN(:, dipoleLoc ), MEG.chanlocs, 'numcontour',  0, 'electrodes', 'off', 'shading', 'interp');
title('Signal dipole projection'), colormap jet

% plot the data from one channel
subplot(212), hold on
% SPW-Rs
plot( MEG.times, .35+ dipole_data(dipoleLoc ,:) / norm(dipole_data(dipoleLoc ,:) ), 'r', 'linew', 1)

% MEG sensor at the event location
MEG_ch = 41; % nearest MEG sensor to ripple
plot( MEG.times, .15 + MEG.data(MEG_ch, :) / norm( MEG.data(MEG_ch, : ) ), 'k', 'linew', 1)
plot( [ .5  .5 ], get(gca,'ylim'), 'k--', 'HandleVisibility', 'off');
xlabel('Time (sec)'), ylabel('(Amplitude)')
legend({'Target Signal (Truth)'; 'Nearest MEG Sensor'})
set(gca,'ytick',[])



%% ----------------------------- %%
%    Ripple reconstruction via PCA    %
%%% --------------------------- %%%

% This code cell demonstrates that PCA is unable
% recover the simulated dipole signal.

% Epoching the target data
tmpData = MEG.data( :, st_time: ed_time);

% Create covariance matrix 4 PCA
cov4PCA = cov(tmpData'); % (channel by channel)

% PCA
[evecs, evals] = eig( cov4PCA );

% sort eigenvalues/vectors
[evals, sidx] = sort( diag(evals), 'descend');
evecs = evecs( :, sidx);

% plot EigenSpectrum
figure(2), clf
subplot(231)
plot(evals, 'go-','markerfacecolor','b','linew', 5, 'markersize', 10)
axis square
set(gca,'xlim',[0 15])
title('PCA eigenvalues')
xlabel('Components '), ylabel('Power ratio (\lambda)')

% Create PC time series (i.e., component) 
comp_ts = evecs( :, 1)'*MEG.data;

% normalize time series (for visualization)
dipl_ts    =  dipole_data(dipoleLoc,:) / norm(dipole_data(dipoleLoc,:));
comp_ts =  comp_ts / norm(comp_ts);
chan_ts  =  MEG.data(MEG_ch ,:) / norm(MEG.data(MEG_ch ,:));

% plot the time series
subplot(212), hold on
plot( MEG.times, .35+dipl_ts, 'r', 'linew', 1)
plot( MEG.times, .15+chan_ts, 'k')
plot( MEG.times, -.005+ comp_ts, 'g')
legend({'Truth';'MEG channel';'PCA time series'})
set(gca,'ytick',[])
xlabel('Time (sec)')

%%% spatial filter forward model via PCA
filt_topo = evecs( :, 1); % using the 1st eigenvector

% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection to be positive.
[~, se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));

% plot the maps
subplot(232)
topoplotIndie( lf.GainN(:,dipoleLoc), MEG.chanlocs, 'numcontour', 0, 'electrodes', 'off', 'shading', 'interp');
title('Truth topomap'), colormap jet

subplot(233)
topoplotIndie(filt_topo, MEG.chanlocs,'electrodes','off','numcontour', 0);
title('PCA forward model'), colormap jet


%% ----------------------------- %%
%    Ripple detection via GED    %
%%% --------------------------- %%%

%%% STEP 1 : Create covariance matrices 4 GED
% Target covariance matrix (i.e., covT)  using the filter data
% Params 4 band-pass filter 
peakfreq = frex; % "ripple" (Hz)
fwhm      = 20; % full-width at half-maximum around the ripple peak

% Do filtering via filterFGx.m
[ filtdata, empVals ] = filterFGx( MEG.data, MEG.srate, peakfreq, fwhm, 1 ); 

% Epoching Data consisting of ripple
tmpData4T = filtdata( :, st_time: ed_time);

% Calc covT (channel by channel)
covT = cov( tmpData4T' ); 


%%% STEP 2 : Create Control covariance matrix (i.e., covC) using the raw data
% Epoching the raw events-related data 
tmpData4C = MEG.data( :, : );
% tmpData4C = MEG.data(:, st_time: ed_time);

% Calc covC (channel by channel)
covC = cov( tmpData4C' ); 


%%% STEP 3 : Shrinkage regulariation 4 covC
% parameters
g = .01;

% calc param "a"
a = mean( eig(covC) );

% new covC
covC_SR = (1-g)*covC + g*a*eye(64); 

%%% plot the two covariance matrices
figure(3), clf

% Target covariance matrix
subplot(141)
imagesc(covT), colormap jet
title('covT')
axis square, %set(gca,'clim',[-1 1]*1e6)
xticks([]); yticks([]); 

% Control covariance matrix
subplot(142)
imagesc(covC), colormap jet
title('covC')
axis square, 
xticks([]); yticks([]); 

% Control covariance matrix with SR
subplot(143)
imagesc(covC_SR), colormap jet
title('Regularized covC via SR')
axis square, 
xticks([]); yticks([]); 

% R^{-1}S
subplot(144)
imagesc(inv(covC)*covT)
title('C^-^1T matrix')
axis square, 
xticks([]); yticks([]); 


%%% STEP 4 : Generalized eigendecomposition (GED)
[evecs, evals] = eig(covT, covC_SR);

% sort eigenvalues/vectors
[evals, sidx] = sort(diag(evals),'descend');
evecs = evecs( :, sidx);

% plot the eigenspectrum
figure(4), clf
subplot(231)
plot(evals, 'ro-','markerfacecolor','b','linew', 5, 'markersize', 10)
axis square
set(gca,'xlim',[0 15])
title('GED eigenvalues')
xlabel('Components'), ylabel('Power ratio (\lambda)')

% GED component as time series 
comp_ts = evecs( :, 1)'*MEG.data; % use the 1st eigenvector

%%% plot for comparison
% normalize time series (for visualization)
dipl_ts = dipole_data(dipoleLoc, :) / norm(dipole_data(dipoleLoc, :));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = MEG.data(MEG_ch,:) / norm(MEG.data(MEG_ch,:));

% plot the time series
subplot(212), hold on
plot( MEG.times, .35+dipl_ts, 'r')
plot( MEG.times, .15+chan_ts, 'k')
plot( MEG.times, -.04+ comp_ts, 'r' )
legend({'Truth';'MEG channel';'GED time series'})
set(gca,'ytick',[])
xlabel('Time (sec)')

%%% spatial filter forward model
% Obtained by passing the covariance matrix through the filter.
filt_topo = covT*evecs(:, 1);

% Eigenvector sign uncertainty can cause a sign-flip, which is corrected for by 
% forcing the largest-magnitude projection sensor to be positive.
[~,se] = max(abs( filt_topo ));
filt_topo = filt_topo * sign(filt_topo(se));

% plot the maps
subplot(232)
topoplotIndie( lf.GainN(:, dipoleLoc), MEG.chanlocs,'numcontour', 0, 'electrodes','off','shading','interp');
title('Truth topomap'), colormap jet

subplot(233)
topoplotIndie( filt_topo, MEG.chanlocs,'electrodes','off','numcontour',0);
title('GED forward model'), colormap jet


%% ----------------------------- %%
%     Ripple reconstruction via ICA     %
%%% --------------------------- %%%

% NOTE: This cell computes ICA based on the jade algorithm (i.e., deterministic algorithms). 
% Make sure the jader() function is in your MATLAB path
% See also:  Rutledge, Douglas N., and D. Jouan-Rimbaud Bouveresse. 
% "Independent components analysis with the JADE algorithm." TrAC Trends in Analytical Chemistry 50 (2013): 22-32.

% Compute ICA via jader
% Epoching the target data
% tmpData = MEG.data( :, st_time: ed_time);
ivecs = jader(tmpData, 40); % It takes some minutes 
ic_scores = ivecs*MEG.data;
icmaps = pinv(ivecs');
evals = diag(icmaps*icmaps');

% plot the IC energy
figure(5), clf
subplot(231)
plot(evals, 'bo-','markerfacecolor','k','linew', 5, 'markersize', 10)
axis square
set(gca,'xlim',[0 15])
title('ICA')
xlabel('Components'), ylabel('IC energy')

% component time series is eigenvector as spatial filter for data
comp_ts = ic_scores(1, :); 

% plot for comparison
% normalize time series (for visualization)
dipl_ts = dipole_data(dipoleLoc,:) / norm(dipole_data(dipoleLoc,:));
comp_ts = comp_ts / norm(comp_ts);
chan_ts = MEG.data(MEG_ch,:) / norm(MEG.data(MEG_ch,:));

% plot the time series
subplot(212), hold on
plot( MEG.times, .35+dipl_ts, 'r')
plot( MEG.times, .15+chan_ts, 'k')
plot( MEG.times, -.04+ comp_ts, 'b' )
legend({'Truth';'MEG channel';'ICA time series'})
set(gca,'ytick',[])
xlabel('Time (sec)')

% plot the maps
subplot(232)
topoplotIndie(lf.GainN(:,dipoleLoc), MEG.chanlocs,'numcontour', 0,'electrodes','off','shading','interp');
title('Truth topomap'), colormap jet

subplot(233)
topoplotIndie(icmaps( 1, : ), MEG.chanlocs,'electrodes','off','numcontour', 0);
title('ICA forward model')


%% done.


