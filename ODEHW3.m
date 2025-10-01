%% ====================== RING N+1 N-BODY DEMO ==========================
% N equally spaced masses m on a ring of radius R around a central mass M.
% Simulate in inertial frame and in a frame rotating at Omega*ẑ.
% Velocity-Verlet (symplectic) time stepping. Fully 3D state.
% -----------------------------------------------------------------------

clear; clc; close all; rng(1);                % fixed seed for reproducibility

%% -------------------- User-set base parameters -------------------------
G      = 1.0;        % gravitational constant
N      = 4;          % number of small bodies on the ring
m      = 1.0;        % mass of each small body
M_over_m = 20;       % ratio M/m to start with
M      = M_over_m * m;
R      = 1.0;        % ring radius
eps_soft   = 1e-4*R; % Plummer softening (set 0 for exact Newtonian if you like)
sigma_frac = 0.00;   % random velocity std as a fraction of v_ring
t_orbits   = 70;     % how long to run (in orbital periods)
dt_per_orbit = 10000; % timesteps per orbit (2000–5000 good)

%% ----------------- Build initial conditions (inertial) -----------------
[pos0, vel0, masses, idxC, idxRing] = init_ring_ic(N, R, M, m);

% Equilibrium Omega from *one ring body* including self-gravity
[Omega, v_ring] = equilibrium_omega(pos0, masses, idxRing(1), G, eps_soft);

% Sanity: force direction & balance check
a0   = grav_accel(pos0, masses, G, eps_soft);
rhat = pos0(idxRing(1),:) / norm(pos0(idxRing(1),:));
ang  = acosd( dot(a0(idxRing(1),:), -rhat) / norm(a0(idxRing(1),:)) );
fprintf('Angle between net force and inward radial: %.6g deg\n', ang);
fprintf('Check balance: a_in/R = %.6g vs Omega^2 = %.6g\n', ...
        (-dot(a0(idxRing(1),:), rhat))/R, Omega^2);

% Random 3-D Gaussian kicks (can be zero)
vrand = sigma_frac * v_ring * randn(numel(idxRing), 3);

% Base tangential velocities for circular motion (inertial frame)
vel0(idxRing, :) = tangential_dirs(pos0(idxRing,:)) * v_ring + vrand;

% Zero total linear momentum by adjusting the central mass velocity
P = sum(masses .* vel0, 1);
vel0(idxC, :) = vel0(idxC, :) - P / masses(idxC);

%% ------------------------- Time parameters -----------------------------
Torb  = 2*pi / Omega;                 % orbital period
dt    = Torb / dt_per_orbit;
tEnd  = t_orbits * Torb;
nSteps = ceil(tEnd/dt);
fprintf('Equilibrium Omega = %.6g, T = %.6g, dt = %.6g, steps = %d\n', Omega, Torb, dt, nSteps);

%% ------------- Inertial-frame integration (Velocity-Verlet) -----------
accFun_inertial = @(r, v, t) grav_accel(r, masses, G, eps_soft);
[ts_I, R_I, V_I, E_I] = vv_integrate(pos0, vel0, accFun_inertial, dt, tEnd, ...
                                     @(r,v) energy_inertial(r,v,masses,G,eps_soft));

% Inertial COM-frame plot (removes any bulk drift)
COM_I   = squeeze(sum(R_I .* reshape(masses,1,[],1), 2) / sum(masses));
R_I_COM = R_I - reshape(COM_I, [], 1, 3);
figure; hold on; axis equal; grid on; view(3);
for k = 1:size(R_I_COM,2), plot3(R_I_COM(:,k,1), R_I_COM(:,k,2), R_I_COM(:,k,3), '-'); end
xlabel('x'), ylabel('y'), zlabel('z'); title('Inertial trajectories in the COM frame');

%% -------- Rotating-frame initial conditions & integration --------------
OmegaVec = [0 0 Omega];

% Rotating-frame positions equal to inertial positions at t=0
pos0_R = pos0;

% Rotating-frame velocities: EXACT mapping v_rot = v_inertial - Omega x r
vrot0 = vel0 - cross(repmat(OmegaVec,numel(masses),1), pos0_R, 2);

% Sanity: mapping back to inertial at t=0 must match vel0
v_inertial_from_rot0 = vrot0 + cross(repmat(OmegaVec,numel(masses),1), pos0_R, 2);
fprintf('Init velocity mismatch (RMS): %.3e\n', ...
    sqrt(mean(sum((v_inertial_from_rot0 - vel0).^2,2))));

% (Optional) Ensure total *inertial* momentum is zero using vrot0 + Ω×r
Pinert0 = sum(masses .* (vrot0 + cross(repmat(OmegaVec,numel(masses),1),pos0_R,2)), 1);
vrot0(idxC,:) = vrot0(idxC,:) - Pinert0/masses(idxC);

accFun_rot = @(r, v, t) grav_accel(r, masses, G, eps_soft) ...
                        - 2*cross(repmat(OmegaVec,size(r,1),1), v, 2) ...
                        - cross(repmat(OmegaVec,size(r,1),1), cross(repmat(OmegaVec,size(r,1),1), r, 2), 2);

% We'll still monitor inertial energy (transform velocities)
Efun_rot = @(r,v) energy_inertial(r, v + cross(repmat(OmegaVec,size(r,1),1), r, 2), masses, G, eps_soft);

[ts_R, R_R, V_R, E_R_inertial] = vv_integrate(pos0_R, vrot0, accFun_rot, dt, tEnd, Efun_rot);

%% ------------------- Compare inertial-vs-rotating runs -----------------
% Transform rotating-frame states to inertial for direct comparison.
[R_R_inertial, V_R_inertial] = rot_to_inertial(R_R, V_R, OmegaVec, ts_R);

% Overlay (solid=inertial, dashed=rotating->inertial)
figure; hold on; axis equal; grid on; view(3);
for k = 1:size(R_I,2),            plot3(R_I(:,k,1),            R_I(:,k,2),            R_I(:,k,3),            '-',  'LineWidth',0.8); end
for k = 1:size(R_R_inertial,2),   plot3(R_R_inertial(:,k,1),   R_R_inertial(:,k,2),   R_R_inertial(:,k,3),   '--', 'LineWidth',0.8); end
xlabel('x'); ylabel('y'); zlabel('z');
title('Fig. 2 — Rotating-frame solution (converted) overlaid on inertial trajectories');
legend('Inertial','Rotating→Inertial');

% Quantify overlap error (should be tiny when dt is small)
Rr = interp1(ts_R, reshape(R_R_inertial, numel(ts_R), []), ts_I, 'linear');
Rr = reshape(Rr, size(R_I));
d  = sqrt(sum((Rr - R_I).^2, 3));        % distance per body
fprintf('Median overlap error: %.3e R, 95th %%: %.3e R\n', ...
    median(d(:))/R, prctile(d(:),95)/R);

% Diagnostics: radial dispersion (std radii), vertical RMS, min separation.
[dispI, zI, mindI] = ring_diagnostics(R_I, idxRing);
[dispR, zR, mindR] = ring_diagnostics(R_R_inertial, idxRing);

% Energy vs time (both runs use inertial energy definition)
figure; plot(ts_I, E_I, 'LineWidth',1.2); hold on;
plot(ts_R, E_R_inertial, '--', 'LineWidth',1.2);
xlabel('t'); ylabel('Total energy (inertial)'); legend('Inertial','Rotating'); grid on;
title('Energy conservation check');

% Radial dispersion
figure; plot(ts_I, dispI, 'LineWidth',1.2); hold on;
plot(ts_R, dispR, '--', 'LineWidth',1.2);
xlabel('t'); ylabel('Std radius of ring'); legend('Inertial','Rotating'); grid on;
title('Ring radial dispersion');

% Out-of-plane (z RMS of ring)
figure; plot(ts_I, zI, 'LineWidth',1.2); hold on;
plot(ts_R, zR, '--', 'LineWidth',1.2);
xlabel('t'); ylabel('RMS(z) of ring'); legend('Inertial','Rotating'); grid on;
title('Out-of-plane growth');

% Minimum inter-particle distance
figure; plot(ts_I, mindI, 'LineWidth',1.2); hold on;
plot(ts_R, mindR, '--', 'LineWidth',1.2);
xlabel('t'); ylabel('Min pair distance'); legend('Inertial','Rotating'); grid on;
title('Collision proximity');

% Quick 3D trajectory snapshot (inertial run)
figure; hold on; axis equal; grid on; view(3);
plot3(R_I(:,idxC,1), R_I(:,idxC,2), R_I(:,idxC,3), 'k-', 'LineWidth',1.2);
for k = idxRing, plot3(R_I(:,k,1), R_I(:,k,2), R_I(:,k,3), '-', 'LineWidth', 0.7); end
xlabel('x'), ylabel('y'), zlabel('z'); title('Inertial trajectories (3D)');

%% ------------------------- dt halving test -----------------------------
dt2 = dt/4; tEnd2 = tEnd;
[ts2, R2, V2, E2] = vv_integrate(pos0, vel0, accFun_inertial, dt2, tEnd2, ...
                                 @(r,v) energy_inertial(r,v,masses,G,eps_soft));
% Compare trajectories on same times
R_interp = interp1(ts2, reshape(R2, numel(ts2), []), ts_I, 'linear');
R_interp = reshape(R_interp, size(R_I));
pos_err = sqrt(mean(sum((R_interp - R_I).^2, 3), 2));   % RMS over bodies
figure; plot(ts_I, pos_err, 'LineWidth',1.2); grid on;
xlabel('t'); ylabel('RMS position difference (dt vs dt/2)');
title('Time-step halving consistency');

%% ------------------ Empirical stability vs M/m sweep -------------------
Ms_over_m = [0, 0.5, 1, 3, 10, 30, 100];
Tprobe_orbits = 20;   % shorter probe for a quick scan
metric = zeros(size(Ms_over_m));
for iM = 1:numel(Ms_over_m)
    Mtest = Ms_over_m(iM)*m;
    [pos0S, vel0S, mS, idxC2, idxRing2] = init_ring_ic(N, R, Mtest, m);
    [OmegaS, vS] = equilibrium_omega(pos0S, mS, idxRing2(1), G, eps_soft);
    TorbS = 2*pi/OmegaS; dtS = TorbS/dt_per_orbit; tEndS = Tprobe_orbits*TorbS;

    vr = sigma_frac*vS*randn(numel(idxRing2),3);
    vel0S(idxRing2,:) = tangential_dirs(pos0S(idxRing2,:))*vS + vr;

    P0 = sum(mS.*vel0S,1);               % zero total momentum
    vel0S(idxC2,:) = vel0S(idxC2,:) - P0/mS(idxC2);

    [tsS, RS, VS] = vv_integrate(pos0S, vel0S, @(r,v,t) grav_accel(r,mS,G,eps_soft), dtS, tEndS);
    [dispS, zS, ~] = ring_diagnostics(RS, idxRing2);

    d0 = max(dispS(1), 1e-12);
    metric(iM) = dispS(end) / d0;        % >~1 means growth (instability)
end

figure; semilogy(Ms_over_m, metric, 'o-', 'LineWidth',1.2, 'MarkerSize',6); grid on;
xlabel('M/m'); ylabel('Radial dispersion growth factor');
title('Empirical ring stability vs mass ratio');

disp('Done. See figures for energy, dispersion, dt test, and stability sweep.');

%% ============================ FUNCTIONS ================================
function [pos0, vel0, masses, idxC, idxRing] = init_ring_ic(N, R, M, m)
    nBodies = N + 1;
    masses = [M; m*ones(N,1)];
    pos0    = zeros(nBodies, 3);
    theta   = (0:N-1)' * 2*pi/N;
    pos0(2:end,1) = R*cos(theta);
    pos0(2:end,2) = R*sin(theta);
    vel0    = zeros(nBodies, 3);
    idxC    = 1;
    idxRing = 2:nBodies;
end

function [Omega, v_ring] = equilibrium_omega(pos, masses, iRing, G, eps_soft)
    r = pos(iRing,:);
    R = norm(r(1:2));
    a = grav_accel(pos, masses, G, eps_soft);
    er = r / R;
    a_inward = -dot(a(iRing,:), er);
    Omega = sqrt(max(a_inward,0) / R);
    v_ring = Omega * R;
end

function T = tangential_dirs(ringXY)
    x = ringXY(:,1); y = ringXY(:,2);
    R = hypot(x,y);
    T = [ -y./R, x./R, zeros(size(R)) ];
end

function a = grav_accel(r, masses, G, eps_soft)
    n = size(r,1);
    a = zeros(n,3);
    for i = 1:n
        d = r - r(i,:);
        dist2 = sum(d.^2,2) + eps_soft^2;
        invR3 = 1 ./ (dist2 .* sqrt(dist2));
        invR3(i) = 0;                       % no self-force
        a(i,:) = G * sum( (masses .* invR3) .* d, 1 );
    end
end

function [ts, R, V, E] = vv_integrate(r0, v0, accFun, dt, tEnd, Efun)
    n = size(r0,1);
    nSteps = ceil(tEnd/dt);
    R = zeros(nSteps+1, n, 3);
    V = zeros(nSteps+1, n, 3);
    ts = (0:nSteps)'*dt;
    R(1,:,:) = r0; V(1,:,:) = v0;
    a = accFun(r0, v0, 0);
    if nargout > 3 && ~isempty(Efun), E = zeros(nSteps+1,1); E(1) = Efun(r0, v0); else, E = []; end
    for k = 1:nSteps
        v_half = squeeze(V(k,:,:)) + 0.5*dt*a;
        r_new  = squeeze(R(k,:,:)) + dt*v_half;
        a_new  = accFun(r_new, v_half, ts(k+1));
        v_new  = v_half + 0.5*dt*a_new;
        R(k+1,:,:) = r_new; V(k+1,:,:) = v_new; a = a_new;
        if ~isempty(E), E(k+1) = Efun(r_new, v_new); end
    end
end

function E = energy_inertial(r, v, masses, G, eps_soft)
    K = 0.5 * sum(masses .* sum(v.^2, 2));
    U = 0; n = size(r,1);
    for i = 1:n-1
        d = r(i+1:end,:) - r(i,:);
        dist = sqrt(sum(d.^2,2) + eps_soft^2);
        U = U - G * masses(i) * sum(masses(i+1:end) ./ dist);
    end
    E = K + U;
end

function [R_inertial, V_inertial] = rot_to_inertial(RR, VR, OmegaVec, t)
    nSteps = numel(t); n = size(RR,2);
    R_inertial = zeros(size(RR)); V_inertial = zeros(size(VR));
    Omega = OmegaVec(3);
    for k = 1:nSteps
        th = Omega * t(k); c = cos(th); s = sin(th);
        Rmat = [c -s 0; s c 0; 0 0 1];
        r_k = squeeze(RR(k,:,:));
        v_k = squeeze(VR(k,:,:));
        wcr = cross(repmat(OmegaVec,n,1), r_k, 2);
        R_inertial(k,:,:) = (Rmat * r_k.').';
        V_inertial(k,:,:) = (Rmat * (v_k + wcr).').';
    end
end

function [radStd, zRMS, minD] = ring_diagnostics(R, idxRing)
    xy = R(:,:,1:2);
    rr = sqrt(sum(xy.^2, 3));
    radStd = std(rr(:, idxRing), 0, 2);
    zRMS   = sqrt(mean(R(:, idxRing, 3).^2, 2));
    T = size(R,1); B = size(R,2); minD = zeros(T,1);
    for k = 1:T
        X = squeeze(R(k,:,:)); S = sum(X.^2, 2);
        D2 = S + S' - 2*(X*X.'); D2 = max(D2, 0);
        D2(1:B+1:end) = inf; minD(k) = sqrt(min(D2(:)));
    end
end
