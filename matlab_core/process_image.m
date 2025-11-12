function result = process_image(image_path)
% Simple example MATLAB function for testing from Python via matlab.engine
% Accepts a file path (string), reads the image and returns a string result.
%
% Usage in Python (matlab.engine):
%   res = eng.feval('process_image', 'C:/path/to/img.jpg', nargout=1)

try
    img = imread(image_path);
    [~, name, ext] = fileparts(image_path);
    % Example: return image size and file name
    s = size(img);
    result = sprintf('Processed %s%s - size: %dx%dx%d', name, ext, s(1), s(2), s(3));
catch ME
    % Return error message as string so Python can get it
    result = ['ERROR: ' ME.message];
end
end
