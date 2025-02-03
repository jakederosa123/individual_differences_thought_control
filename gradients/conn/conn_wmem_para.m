% Add paths for SPM and CONN toolboxes
spm_path = '/projects/ics/software/spm/spm12_v6470';
conn_path = '/projects/ics/software/conn21a';
addpath(spm_path);
addpath(conn_path);

% Path to the subjects directory
subjects_dir = '/pl/active/banich/studies/wmem/fmri/subjects';

% Get a list of all subjects in the directory
subject_folders = dir(fullfile(subjects_dir, '*'));
subject_folders = subject_folders([subject_folders.isdir]);
subject_folders = subject_folders(~ismember({subject_folders.name}, {'.', '..'}));

% Set up parallel processing
RUNPARALLEL = true; % run in parallel using computer cluster
NSUBJECTS = []; % leave empty for all subjects
NJOBS = []; % number of parallel jobs to submit

% Initialize batch structure
batch = struct();
batch.filename = fullfile(subjects_dir, 'conn_wmem.mat');

% Initialize subjects list
valid_subjects = {};
valid_functionals = {};
valid_structurals = {};

% Loop over subjects to check for required files and set up the batch for each valid subject
for nsub = 1:numel(subject_folders)
    subject_folder = fullfile(subjects_dir, subject_folders(nsub).name);
    subject_id = subject_folders(nsub).name; % Get the actual subject ID
    fprintf('Checking subject %s\n', subject_folder);
    
    % Check if all required functional files exist
    required_functionals = {
        'restrun1.nii.gz', 'restrun2.nii.gz', 'wmemrun1.nii.gz', ...
        'wmemrun2.nii.gz', 'wmemrun3.nii.gz', 'wmemrun4.nii.gz', ...
        'wmemrun5.nii.gz', 'wmemrun6.nii.gz'
    };
    all_functionals_exist = all(cellfun(@(x) exist(fullfile(subject_folder, x), 'file'), required_functionals));
    
    % Check if 't1w.nii.gz' exists
    t1w_exists = exist(fullfile(subject_folder, 't1w.nii.gz'), 'file');
    
    % If the required files exist, add the subject to the valid subjects list
    if all_functionals_exist && t1w_exists
        valid_subjects{end+1} = subject_id; % Store the actual subject ID
        
        functionals = cellfun(@(x) fullfile(subject_folder, x), required_functionals, 'UniformOutput', false);
        
        valid_functionals{end+1} = functionals;
        valid_structurals{end+1} = fullfile(subject_folder, 't1w.nii.gz');
    else
        fprintf('Skipping subject %s (missing required files)\n', subject_folder);
    end
end

% If no valid subjects found, exit
if isempty(valid_subjects)
    error('No valid subjects found with the required files.');
end

% Setup: Structural and Functional Files
batch.Setup.isnew = 1;
batch.Setup.nsubjects = numel(valid_subjects);
batch.Setup.RT = 0.46; % Repetition time (in seconds)

% Initialize the functional and structural files
for nsub = 1:numel(valid_subjects)
    batch.Setup.functionals{nsub} = valid_functionals{nsub};
    batch.Setup.structurals{nsub} = valid_structurals{nsub};
end

% Atlas file for the 360 glasser parcels
batch.Setup.rois.names = {'Glasser360'};
batch.Setup.rois.files = {fullfile('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/glasser360MNI.nii.gz')};
batch.Setup.rois.multiplelabels = 1; % Indicate multiple labels for Glasser atlas

% Import conditions from the CSV file
condition_file = fullfile('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/conn/sub-001_ses-001_all_tasks_events.csv');
batch.Setup.conditions.importfile = condition_file;

% Preprocessing
batch.Setup.preprocessing.steps = {'default_mni'};
batch.Setup.preprocessing.sliceorder = 'interleaved (Siemens)'; % Correct slice order for Siemens scanner
batch.Setup.preprocessing.fwhm = 12; % Smoothing kernel (in mm)
batch.Setup.preprocessing.art_thresholds = [3 .5]; % Conservative thresholds for motion (mm) and intensity (global signal z-value)

% Analysis options
batch.Setup.analyses = 1; % 1: ROI-to-ROI

% Setup: Run the Setup
batch.Setup.done = 1; % Make sure this line is active to complete the setup step
batch.Setup.overwrite = 'Yes';

% Denoising
batch.Denoising.filter = [0.01, inf]; % Frequency filter (band-pass values, in Hz)
batch.Denoising.detrending = 1; % Linear detrending (0 = none, 1 = linear, 2 = quadratic)
batch.Denoising.done = 1; % Indicate that denoising is complete
batch.Denoising.overwrite = 'Yes';

% Initialize Analysis structure correctly
batch.Analysis = struct();

% First-level analysis ROI-to-ROI
batch.Analysis.type = 1; % Analysis type 1 for ROI-to-ROI
batch.Analysis.measure = 1; % ROI-to-ROI connectivity (RRC)
batch.Analysis.weight = 2; % Default weighting for resting state
batch.Analysis.sources = {'Glasser360'}; % Using the defined ROIs
batch.Analysis.conditions = {'rest', 'clear', 'maintain', 'replace', 'suppress'}; % Ensure analysis for all conditions

% Indicate completion of the Analysis step
batch.Analysis.done = 1; 
batch.Analysis.overwrite = 'Yes';

% Set up parallel processing if required
if RUNPARALLEL
    batch.parallel.N = NJOBS; % number of parallel processing batch jobs
end

% Save the actual subject IDs for reference
batch.Setup.subjectIDs = valid_subjects;

% Run the batch
conn_batch(batch);
