
classdef Adult < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = Adult(nTr , nTe, outputFormat, shuffleTraining, shuffleTest, shuffleAll)
            
            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
            display('This implementation of the Adult dataset considers all the available attributes, (d = 123)');
            obj.d = 123;        % Fixed size for the full dataset
            data = load('adult.mat');
            
            % Fix dimensionality issues
            tesz = size(data.testing_vectors,2);
            if tesz < obj.d
                data.testing_vectors = [data.testing_vectors , zeros(size(data.testing_vectors,1), obj.d - tesz)];
            end
            
            trsz = size(data.training_vectors,2);
            if trsz < obj.d
                data.testing_vectors = [data.testing_vectors , zeros(size(data.testing_vectors,1), obj.d - trsz)];
            end            
            
            obj.X = [data.training_vectors ; data.testing_vectors];
            obj.X = obj.scale(obj.X);
            obj.Y = [data.training_labels ; data.testing_labels];
                            
            obj.nTrTot = size(data.training_labels,1);
            obj.nTeTot = size(data.testing_labels,1);
            obj.n = size(obj.Y , 1);

            obj.t = 2;
                
            if nargin == 0

                obj.nTr = obj.nTrTot;
                obj.nTe = obj.nTeTot;
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            elseif (nargin >1)
                
                if (nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                         
                obj.trainIdx = 1:obj.nTr;          
                
                obj.testIdx = obj.nTrTot + 1:obj.nTe + obj.nTrTot;
            end
            
            % Shuffling
            obj.shuffleTraining = shuffleTraining;
            if shuffleTraining == 1
                obj.shuffleTrainIdx();
            end
            
            obj.shuffleTest = shuffleTest;
            if shuffleTest == 1
                obj.shuffleTestIdx();
            end
            
            obj.shuffleAll = shuffleAll;
            if shuffleAll == 1
                obj.shuffleAllIdx();
            end           
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            if strcmp(obj.outputFormat, 'plusMinusOne')
                obj.Y = obj.Y*2-1;
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                obj.Y = obj.Y*2-1;
            end
            
            obj.problemType = 'classification';
        end
        
        % Checks if matrix Y contains real values. Useful for
        % discriminating between classification and regression, or between
        % predicted scores and classes
        function res = hasRealValues(obj , M)
        
            res = false;
            for i = 1:size(M,1)
                for j = 1:size(M,2)
                    if mod(M(i,j),1) ~= 0
                        res = true;
                    end
                end
            end
        end
        
        % Compute predictions matrix from real-valued scores matrix
        function Ypred = scoresToClasses(obj , Yscores)    
            
            if strcmp(obj.outputFormat, 'zeroOne')
                Ypred = zeros(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusMinusOne')
                Ypred = -1 * ones(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                Ypred = -1/(obj.t - 1) * ones(size(Yscores));
            end
            
%             for i = 1:size(Ypred,1)
%                 if Yscores(i) > 0
%                     Ypred(i) = 1;
%                 end
%             end
            
            Ypred(Yscores>0) = 1;
        end
            
        % Compute performance measure on the given outputs according to the specified loss
        function perf = performanceMeasure(obj , Y , Yscores , varargin)
            
            % Check if Ypred is real-valued. If yes, convert it.
            Ypred = obj.scoresToClasses(Yscores);

            perf = obj.lossFunction(Y, Yscores, Ypred);
            
        end
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)
            
            mx = max(max(M));
            mn = min(min(M));
            
            Ms = ((M + abs(mn)) / (mx - mn)) * 2 - 1;
            
        end
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef
