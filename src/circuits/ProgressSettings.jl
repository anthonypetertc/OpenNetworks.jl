module ProgressSettings
export default_progress_kwargs

default_progress_kwargs = Dict{Symbol,Any}(
    :desc => "Applying circuit...", :dt => 1.0, :enabled => true
)

end; # module
