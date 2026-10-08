# gui/server.jl  —  local web server for the shimming-pipeline GUI (STEP 3 shell)
#
#   julia gui/server.jl              # starts the server, opens the browser
#
# Serves gui/app.html and a small JSON API over Backend (gui/backend.jl):
#   GET  /api/config        → current config.toml as JSON
#   POST /api/config        → apply edits (validated) to config.toml
#   GET  /api/measured      → measured CSVs + the currently selected one
#   POST /api/import        → {src}   copy a CSV into the data folder
#   GET  /api/osii          → OSII shim-config CSVs + the currently selected one
#   POST /api/import_osii   → {src}   copy an OSII CSV into data/inputs/OSII_shimming_outputs_toconvert/
#   POST /api/open_folder   → {which} open a project folder in the OS (root, measured, osii, cache, assets, optimizer, viewers, verifier, final)
#   POST /api/run           → {stage} run a stage, streaming its log line-by-line
#
# No build step: app.html is plain HTML/CSS/JS. Needs HTTP + JSON (install_deps.jl).

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          # project lives in the repo root
using HTTP, JSON

include(joinpath(@__DIR__, "backend.jl"))
using .Backend

const PORT     = 8010
const APP_HTML = joinpath(@__DIR__, "app.html")
const PRESETS  = joinpath(@__DIR__, "presets.json")

read_presets() = isfile(PRESETS) ? JSON.parse(read(PRESETS, String)) :
                 Dict("geometry" => Dict(), "stage2" => Dict())

json_resp(x; status = 200) =
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON.json(x))

body_json(req) = isempty(req.body) ? Dict{String,Any}() : JSON.parse(String(req.body))

function handle(req::HTTP.Request)
    m, path = req.method, HTTP.URI(req.target).path
    try
        if m == "GET" && path == "/"
            return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"],
                                 read(APP_HTML, String))

        elseif m == "GET" && path == "/api/config"
            return json_resp(Backend.read_config())

        elseif m == "POST" && path == "/api/config"
            Backend.write_config(body_json(req))
            return json_resp(Dict("ok" => true, "config" => Backend.read_config()))

        elseif m == "GET" && path == "/api/measured"
            cfg = Backend.read_config()
            return json_resp(Dict("files" => Backend.list_measured(),
                                  "selected" => get(cfg, "measured_fieldmap_name", "")))

        elseif m == "POST" && path == "/api/import"
            name = Backend.import_csv(String(body_json(req)["src"]))
            return json_resp(Dict("ok" => true, "name" => name,
                                  "files" => Backend.list_measured()))

        elseif m == "GET" && path == "/api/osii"
            cfg = Backend.read_config()
            return json_resp(Dict("files" => Backend.list_osii(),
                                  "selected" => get(cfg, "osii_input_name", "")))

        elseif m == "POST" && path == "/api/import_osii"
            name = Backend.import_osii(String(body_json(req)["src"]))
            return json_resp(Dict("ok" => true, "name" => name,
                                  "files" => Backend.list_osii()))

        elseif m == "POST" && path == "/api/stop"
            return json_resp(Dict("stopped" => Backend.stop_run()))

        elseif m == "GET" && path == "/api/placement"
            # what the current shim CSV ACTUALLY contains, per (ring, tray) — a
            # sparse per-insert or OSII layout fills only some trays of a ring
            p = Backend.placement()
            return json_resp(Dict("exists" => p.exists, "magnets" => p.magnets,
                                  "rings" => [Dict("ring" => r.ring, "trays" => r.trays,
                                                   "magnets" => r.magnets) for r in p.rings]))

        elseif m == "GET" && path == "/api/presets"
            return json_resp(read_presets())

        elseif m == "POST" && path == "/api/presets"
            b = body_json(req)                       # {category, name, values}
            cat, name, vals = b["category"], b["name"], b["values"]
            all = read_presets()
            haskey(all, cat) || (all[cat] = Dict())
            all[cat][name] = vals
            open(PRESETS, "w") do io; JSON.print(io, all, 2); end
            return json_resp(Dict("ok" => true, "presets" => all))

        elseif m == "POST" && path == "/api/open_folder"
            which = get(body_json(req), "which", "root")
            d = Backend.output_dirs()
            path2 = which == "final"     ? d.final :
                    which == "optimizer" ? d.optimizer :
                    which == "verifier"  ? d.verifier :
                    which == "viewers"   ? d.viewers :
                    which == "measured"  ? Backend.MEASURED_DIR :
                    which == "osii"      ? Backend.OSII_DIR :
                    which == "cache"     ? Backend.CACHE_DIR :
                    which == "assets"    ? Backend.ASSETS_DIR : Backend.REPO
            Backend.open_folder(path2)
            return json_resp(Dict("ok" => true, "opened" => path2))

        else
            return HTTP.Response(404, "not found: $m $path")
        end
    catch e
        @error "request failed" method=m path=path exception=(e, catch_backtrace())
        return json_resp(Dict("error" => sprint(showerror, e)); status = 400)
    end
end

# --- streaming: run a stage, pipe its log to the browser line-by-line ---------
const RUNNING = Ref(false)          # one GPU run at a time

function handle_run(stream::HTTP.Stream, req)
    stage = try; String(get(body_json(req), "stage", "")); catch; ""; end
    HTTP.setstatus(stream, 200)
    HTTP.setheader(stream, "Content-Type" => "text/plain; charset=utf-8")
    HTTP.setheader(stream, "Cache-Control" => "no-cache")
    HTTP.startwrite(stream)
    emit(s) = (write(stream, s); flush(stream))
    if RUNNING[]
        emit("✘ a run is already in progress\n__EXIT__ 1\n"); return
    end
    if !haskey(Backend.STAGES, stage)
        emit("✘ unknown stage: $(stage)\n__EXIT__ 1\n"); return
    end
    RUNNING[] = true
    try
        emit("▶ starting $(stage) …\n")
        code = Backend.run_stage(stage; on_output = l -> emit(l * "\n"))
        emit("__EXIT__ $(code)\n")
    catch e
        emit("✘ error: $(sprint(showerror, e))\n__EXIT__ 1\n")
    finally
        RUNNING[] = false
    end
    return
end

# Stream handler: /api/run streams; everything else uses the Request→Response `handle`.
function serve_stream(stream::HTTP.Stream)
    req = stream.message
    req.body = read(stream)
    path = HTTP.URI(req.target).path
    if req.method == "POST" && path == "/api/run"
        return handle_run(stream, req)
    end
    resp = handle(req)
    HTTP.setstatus(stream, resp.status)
    for (k, v) in resp.headers
        HTTP.setheader(stream, k => v)
    end
    HTTP.setheader(stream, "Content-Length" => string(sizeof(resp.body)))
    HTTP.startwrite(stream)
    write(stream, resp.body)
    return
end

function open_browser(url)
    opener = Sys.iswindows() ? `cmd /c start $url` :
             Sys.isapple()   ? `open $url` : `xdg-open $url`
    try; run(opener); catch; end
end

url = "http://localhost:$(PORT)/"
println("Shimming GUI → ", url, "   (Ctrl-C to stop)")
open_browser(url)
HTTP.serve(serve_stream, "127.0.0.1", PORT; stream = true)
