%% WEC-Sim → SQLite export (downsamp + full-res) using only m2g_out

model     = string(m2g_out.model);
sim_id    = string(m2g_out.sim_id);
tbl_down  = "WECSIM_" + model + "_" + sim_id;
tbl_full  = tbl_down + "_full";

% ---------- Downsampled table (time,p,q,base) ----------
t_ds   = m2g_out.Pgrid_ds.Time(:);
p_dsMW = m2g_out.Pgrid_ds.Data(:) / 1e6;   % W → MW
q_dsMW = zeros(size(p_dsMW));
base1  = ones(size(p_dsMW));               % 1.0 MW base

T_down = table(t_ds, p_dsMW, q_dsMW, base1, ...
    'VariableNames', {'time','p','q','base'});

% ---------- Full-res table (time,p,q,base,eta) ----------
t_raw   = m2g_out.Pgrid.Time(:);
p_rawMW = m2g_out.Pgrid.Data(:) / 1e6;
q_rawMW = zeros(size(p_rawMW));
base1r  = ones(size(p_rawMW));

% Align eta to Pgrid time using m2g_out.t_eta / m2g_out.eta
if isfield(m2g_out,'t_eta') && ~isempty(m2g_out.t_eta)
    t_eta = m2g_out.t_eta(:);
    eta   = m2g_out.eta(:);
    if numel(t_eta) ~= numel(t_raw) || any(t_eta ~= t_raw)
        eta_on_power = interp1(t_eta, eta, t_raw, 'linear', 'extrap');
    else
        eta_on_power = eta;
    end
else
    warning('m2g_out.t_eta / m2g_out.eta missing; writing NaN eta.');
    eta_on_power = nan(size(t_raw));
end

T_full = table(t_raw, p_rawMW, q_rawMW, base1r, eta_on_power, ...
    'VariableNames', {'time','p','q','base','eta'});

% ---------- Write both to SQLite ----------
dbfile = fullfile(DB_PATH);
conn   = sqlite(dbfile);

exec(conn, sprintf( ...
    "CREATE TABLE IF NOT EXISTS %s (time FLOAT, p FLOAT, q FLOAT, base FLOAT)", tbl_down));
exec(conn, sprintf( ...
    "CREATE TABLE IF NOT EXISTS %s (time FLOAT, p FLOAT, q FLOAT, base FLOAT, eta FLOAT)", tbl_full));

insert(conn, tbl_down, {'time','p','q','base'}, T_down);
insert(conn, tbl_full, {'time','p','q','base','eta'}, T_full);

close(conn);