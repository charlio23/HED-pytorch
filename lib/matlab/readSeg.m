function [seg] = readSeg(filename)
% function [seg] = reaSeg(filename)
%
% Read a segmentation file.
% Return a segment membership matrix with values [1,k].
%
% Charless Fowlkes <fowlkes@eecs.berkeley.edu>
% David Martin <dmartin@eecs.berkeley.edu>
% January 2003
fid = load(filename);
seg = {};

for i = 1:length(fid.groundTruth)
    seg{i} = fid.groundTruth{i}.Boundaries;
end