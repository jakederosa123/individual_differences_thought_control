% Add paths for SPM and CONN toolboxes
spm_path = '/projects/ics/software/spm/spm12_v6470';
conn_path = '/projects/ics/software/conn21a';
addpath(spm_path);
addpath(conn_path);

% Path to the subjects directory
subjects_dir = '/pl/active/banich/studies/wmem/fmri/subjects';
results_dir = '/pl/active/banich/studies/wmem/fmri/subjects/rest_results'; % Define a results directory

% Get a list of all subjects in the directory
subject_folders = dir(fullfile(subjects_dir, '*'));
subject_folders = subject_folders([subject_folders.isdir]);
subject_folders = subject_folders(~ismember({subject_folders.name}, {'.', '..'}));

% Loop over subjects to check for required files and process each valid subject
for nsub = 1:numel(subject_folders)
    subject_folder = fullfile(subjects_dir, subject_folders(nsub).name);
    subject_id = subject_folders(nsub).name; % Get the actual subject ID
    fprintf('Checking subject %s\n', subject_folder);
    
    % Check if subject has already been processed

    condition_file = fullfile(results_dir, subject_id, '/rest_processing_batch/results/firstlevel/SBC_01/resultsROI_Condition001.mat');
    if exist(condition_file, 'file')
        fprintf('Skipping subject %s (already processed)\n', subject_folder);
        continue;
    end
    
    % Check if all required functional files exist
    required_functionals = {
        'restrun1.nii.gz', 'restrun2.nii.gz'
    };
    all_functionals_exist = all(cellfun(@(x) exist(fullfile(subject_folder, x), 'file'), required_functionals));
    
    % Check if 't1w.nii.gz' exists
    t1w_exists = exist(fullfile(subject_folder, 't1w.nii.gz'), 'file');
    
    % If the required files exist, process the subject
    if all_functionals_exist && t1w_exists
        fprintf('Processing subject %s\n', subject_folder);
        
        % Create a subfolder for each valid subject in the results directory
        subject_results_folder = fullfile(results_dir, subject_id);
        if ~exist(subject_results_folder, 'dir')
            mkdir(subject_results_folder);
        end
        
        % Copy the required files to the subject's results folder
        functionals = cellfun(@(x) fullfile(subject_results_folder, x), required_functionals, 'UniformOutput', false);
        cellfun(@(x) copyfile(fullfile(subject_folder, x), fullfile(subject_results_folder, x)), required_functionals);
        copyfile(fullfile(subject_folder, 't1w.nii.gz'), fullfile(subject_results_folder, 't1w.nii.gz'));
        
        % Initialize batch structure for the current subject
        batch = struct();
        batch.filename = fullfile(subject_results_folder, 'rest_processing_batch.mat'); % Save the project file in the subject's results folder
        batch.Setup = struct();
        batch.Setup.isnew = 1;
        batch.Setup.nsubjects = 1;
        batch.Setup.RT = 0.46; % Repetition time (in seconds)
        batch.Setup.functionals{1} = functionals;
        batch.Setup.structurals{1} = fullfile(subject_results_folder, 't1w.nii.gz');
        batch.Setup.voxelresolution = 1;
        batch.Setup.rois.names = {'Glasser360'};
        batch.Setup.rois.files = {fullfile('/pl/active/banich/studies/wmem/fmri/operation_rsa/grp/glasser360MNI.nii')};
        batch.Setup.rois.multiplelabels = 1; % Indicate multiple labels for Glasser atlas
        
        % Condition for resting state
        batch.Setup.conditions.names = {'rest'}; % Single condition (aggregate across all sessions)
        
        % Check if 'restrun2.nii.gz' exists and set nsessions accordingly
        if exist(fullfile(subject_folder, 'restrun2.nii.gz'), 'file')
            nsessions = 2;
            batch.Setup.functionals{1} = {
                fullfile(subject_results_folder, 'restrun1.nii.gz'),
                fullfile(subject_results_folder, 'restrun2.nii.gz')
            };
        else
            nsessions = 1;
            batch.Setup.functionals{1} = {
                fullfile(subject_results_folder, 'restrun1.nii.gz')
            };
        end
        
        % Initialize conditions
        for ncond = 1:length(batch.Setup.conditions.names)
            for nsub = 1:batch.Setup.nsubjects
                for nses = 1:nsessions
                    batch.Setup.conditions.onsets{ncond}{nsub}{nses} = 0;
                    batch.Setup.conditions.durations{ncond}{nsub}{nses} = inf;
                end
            end
        end
        
        % Preprocessing
        batch.Setup.preprocessing.steps = {'default_mni'};
        batch.Setup.preprocessing.sliceorder = 'interleaved (Siemens)'; % Correct slice order for Siemens scanner
        batch.Setup.preprocessing.fwhm = 8; % Smoothing kernel (in mm)
        batch.Setup.preprocessing.art_thresholds = [3 .5]; % Conservative thresholds for motion (mm) and intensity (global signal z-value)
        
        % Analysis options
        batch.Setup.analyses = 1; % 1: ROI-to-ROI
        batch.Setup.done = 1; % Make sure this line is active to complete the setup step
        batch.Setup.overwrite = 'Yes';
        
        % Denoising
        batch.Denoising = struct();
        batch.Denoising.filter = [0.01, inf]; % Frequency filter (band-pass values, in Hz)
        batch.Denoising.detrending = 1; % Linear detrending (0 = none, 1 = linear, 2 = quadratic)
        batch.Denoising.done = 1; % Indicate that denoising is complete
        batch.Denoising.overwrite = 'Yes';
        
        % Initialize Analysis structure correctly
        batch.Analysis = struct();
        batch.Analysis.type = 1; % Analysis type 1 for ROI-to-ROI
        batch.Analysis.measure = 1; % ROI-to-ROI connectivity (RRC)
        batch.Analysis.weight = 1; % Default weighting for resting state
        batch.Analysis.sources = {'Glasser360'}; % Using the defined ROIs
        batch.Analysis.conditions = {'rest'}; % Ensure analysis for resting state
        batch.Analysis.done = 1; 
        batch.Analysis.overwrite = 'Yes';
        
        % Ensure parallel processing is disabled
        batch.parallel.N = 0;
        batch.parallel.immediate = false;
        batch.parallel.profile = ''; % Set an empty profile to ensure no parallel processing
        
        % Save the actual subject ID for reference
        batch.Setup.subjectIDs = {subject_id};
        
        % Run the batch for the current subject
        conn_batch(batch);
    else
        fprintf('Skipping subject %s (missing required files)\n', subject_folder);
    end
end
