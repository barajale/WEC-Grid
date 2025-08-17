%% WEC-Sim → SQLite export using new WEC-Grid database schema

% Extract simulation metadata from m2g_out
model_type    = char(m2g_out.model);          % Convert to char array for SQLite
sim_duration  = double(m2g_out.simLength);
delta_time    = double(m2g_out.dt);
wave_height   = double(m2g_out.Hs);
wave_period   = double(m2g_out.Tp);
wave_spectrum = char(m2g_out.spectrumType);
wave_class    = char(m2g_out.waveClass);
wave_seed     = int32(m2g_out.seed);

% Generate simulation hash for uniqueness (using char arrays)
sim_hash = sprintf('%s_%.1fm_%.1fs_%d', model_type, wave_height, wave_period, wave_seed);

% ---------- WEC simulation metadata ----------
dbfile = string(DB_PATH);
conn   = sqlite(dbfile);

% Insert WEC simulation record
exec(conn, sprintf(['CREATE TABLE IF NOT EXISTS wec_simulations (', ...
    'wec_sim_id INTEGER PRIMARY KEY AUTOINCREMENT, ', ...
    'model_type TEXT NOT NULL, ', ...
    'sim_duration_sec REAL NOT NULL, ', ...
    'delta_time REAL NOT NULL, ', ...
    'wave_height_m REAL, ', ...
    'wave_period_sec REAL, ', ...
    'wave_spectrum TEXT, ', ...
    'wave_class TEXT, ', ...
    'wave_seed INTEGER, ', ...
    'simulation_hash TEXT, ', ...
    'created_at TEXT DEFAULT CURRENT_TIMESTAMP)']));

% Insert simulation metadata
fprintf('Inserting simulation metadata...\n');
fprintf('  model_type: %s (class: %s)\n', model_type, class(model_type));
fprintf('  wave_spectrum: %s (class: %s)\n', wave_spectrum, class(wave_spectrum));
fprintf('  wave_class: %s (class: %s)\n', wave_class, class(wave_class));
fprintf('  sim_hash: %s (class: %s)\n', sim_hash, class(sim_hash));

insert(conn, 'wec_simulations', ...
    {'model_type', 'sim_duration_sec', 'delta_time', 'wave_height_m', ...
     'wave_period_sec', 'wave_spectrum', 'wave_class', 'wave_seed', 'simulation_hash'}, ...
    {model_type, sim_duration, delta_time, wave_height, ...
     wave_period, wave_spectrum, wave_class, wave_seed, sim_hash});

% Get the wec_sim_id that was just inserted
wec_sim_id_result = fetch(conn, 'SELECT last_insert_rowid() as wec_sim_id');

% Handle different return types from fetch()
if iscell(wec_sim_id_result)
    wec_sim_id = wec_sim_id_result{1};
elseif isnumeric(wec_sim_id_result)
    wec_sim_id = wec_sim_id_result;
elseif istable(wec_sim_id_result)
    wec_sim_id = wec_sim_id_result.wec_sim_id;
else
    % Try direct field access for struct
    wec_sim_id = wec_sim_id_result.wec_sim_id;
end

% ---------- WEC power time-series data ----------
% Extract full-resolution power data
t_raw   = m2g_out.Pgrid.Time(:);
p_raw_w = m2g_out.Pgrid.Data(:);  % Keep in Watts as per schema
q_raw_w = zeros(size(p_raw_w));   % Reactive power (typically zero for WECs)

% Align wave elevation to power time grid
if isfield(m2g_out,'t_eta') && ~isempty(m2g_out.t_eta)
    t_eta = m2g_out.t_eta(:);
    eta   = m2g_out.eta(:);
    if numel(t_eta) ~= numel(t_raw) || any(abs(t_eta - t_raw) > 1e-6)
        eta_aligned = interp1(t_eta, eta, t_raw, 'linear', 'extrap');
    else
        eta_aligned = eta;
    end
else
    warning('m2g_out.t_eta / m2g_out.eta missing; writing NaN wave elevation.');
    eta_aligned = nan(size(t_raw));
end

% Create power results table
exec(conn, sprintf(['CREATE TABLE IF NOT EXISTS wec_power_results (', ...
    'wec_sim_id INTEGER NOT NULL, ', ...
    'time_sec REAL NOT NULL, ', ...
    'device_index INTEGER NOT NULL, ', ...
    'p_w REAL, ', ...
    'q_var REAL, ', ...
    'wave_elevation_m REAL, ', ...
    'PRIMARY KEY (wec_sim_id, time_sec, device_index), ', ...
    'FOREIGN KEY (wec_sim_id) REFERENCES wec_simulations(wec_sim_id) ON DELETE CASCADE)']));

% Prepare data for batch insert (assuming single device for now, device_index = 1)
device_index = ones(size(t_raw));  % Single device
wec_sim_ids = repmat(wec_sim_id, size(t_raw));

T_power = table(wec_sim_ids, t_raw, device_index, p_raw_w, q_raw_w, eta_aligned, ...
    'VariableNames', {'wec_sim_id', 'time_sec', 'device_index', 'p_w', 'q_var', 'wave_elevation_m'});

% Insert power time-series data
insert(conn, 'wec_power_results', ...
    {'wec_sim_id', 'time_sec', 'device_index', 'p_w', 'q_var', 'wave_elevation_m'}, ...
    T_power);

close(conn);

% Return the wec_sim_id to MATLAB workspace for Python to retrieve
wec_sim_id_result = wec_sim_id;
fprintf('WEC-Sim data stored: wec_sim_id = %d, %d time points\n', wec_sim_id, length(t_raw));