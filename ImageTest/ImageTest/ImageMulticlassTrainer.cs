using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Tensorflow;
using Tensorflow.Keras.Engine;

namespace ImageLearning
{
    internal static class ImageMulticlassTrainer
    {

        private static readonly MLContext _mlContext;
        //private static readonly EstimatorChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>> _transformer;
        private static readonly EstimatorChain<KeyToValueMappingTransformer> _transformer;
        private static readonly LbfgsMaximumEntropyMulticlassTrainer _trianer;
        private static PredictionEngine<ImageModel, ImagePrediction> _predictionEngine = default!;
        private static MulticlassPredictionTransformer<MaximumEntropyModelParameters> _model = default!;
        private static ITransformer _transformerChain = default!;
        //private static ITransformer _predictionModel = default;
        private static string _modelPath;
        private static string _chainPath;
        private static readonly string _inceptionTensorFlowModel;
        public static string ImagePath;

        //private static MaximumEntropyModelParameters _preModelParameters;

        private static readonly object _lock = new object();
        static ImageMulticlassTrainer()
        {
            
            ImagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "AssertImages");
            _inceptionTensorFlowModel = Path.Combine(ImagePath, "inception", "tensorflow_inception_graph.pb");
            _mlContext = new MLContext();


            _trianer = _mlContext
                  .MulticlassClassification
                  .Trainers
                  .LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation");


            _transformer = _mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: ImagePath, inputColumnName: nameof(ImageModel.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(_mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(_trianer)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(_mlContext);


           

            _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "image_model.zip");
            _chainPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "image_pipe.zip");
            if (File.Exists(_modelPath))
            {
                ReloadOldModel();
            }

        }

        private static void ReloadOldModel()
        {
            _transformerChain = _mlContext.Model.Load(_chainPath, out _);
            //_predictionModel = _mlContext.Model.Load(_modelPath, out _);
            _model = (MulticlassPredictionTransformer<MaximumEntropyModelParameters>)_mlContext.Model.Load(_modelPath, out _);
        }

        public static void Train(params ImageModel[] images)
        {
            InternalTrain(images);
        }
        public static void Train(IEnumerable<ImageModel> images)
        {
            InternalTrain(images);
        }
        internal static void InternalTrain(IEnumerable<ImageModel> images)
        {
            lock (_mlContext)
            {

                IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);
                if (_model != null)
                {
                    //if exist model, load and train new model.
                    ReloadOldModel();
                    IDataView transformedNewData = _transformerChain.Transform(imageData);
                    _model =_trianer.Fit(transformedNewData, _model.Model);
                }
                else
                {
                    //first fit. get chain and model.
                    var pipe = _transformer.Fit(imageData);
                    _transformerChain = pipe;
                    _model = (MulticlassPredictionTransformer<MaximumEntropyModelParameters>)(pipe.Skip(pipe.Count() - 2).Take(1).First());
                }

                _mlContext.Model.Save(_transformerChain, null, _chainPath);
                _mlContext.Model.Save(_model, null, _modelPath);
            }
            
        }

        public static void Evaluate(params ImageModel[] images)
        {
            InternalEvaluate(images);
        }
        public static void Evaluate(IEnumerable<ImageModel> images)
        {
            InternalEvaluate(images);
        }
        internal static void InternalEvaluate(IEnumerable<ImageModel> images)
        {

            IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);
            IDataView predictions = _model.Transform(imageData);
            MulticlassClassificationMetrics metrics =_mlContext
                .MulticlassClassification
                .Evaluate(predictions,
                            labelColumnName: "LabelKey",
                            predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
           
        }
        public static void Predict(params ImageModel[] images)
        {
            InternalPredictResult(images);
        }
        public static void Predict(IEnumerable<ImageModel> images)
        {
            InternalPredictResult(images);
        }
        public static void InternalPredictResult(IEnumerable<ImageModel> images)
        {

            IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);
            IDataView transformData = _transformerChain.Transform(imageData);
            IDataView predictionData = _model.Transform(transformData);

            var results = _mlContext.Data.CreateEnumerable<ImagePrediction>(predictionData, reuseRowObject: true);
            lock (_lock)
            {
                foreach (var prediction in results)
                {
                    Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predict as: {prediction.PredictedLabelValue} score: {prediction.Score.Max()}");
                }
            }

        }
        /*
       public static void TrainTheBestData(IEnumerable<BcwQuestion> bcwQuestions, int count)
       {

           var differ = new double[count];
           var transformers = new ITransformer[count];
           var data = _mlContext.Data.LoadFromEnumerable(bcwQuestions);

           var result = Parallel.For(0, count, i =>
           {
               (transformers[i], differ[i]) = Evaluate(data);
           });
           while (!result.IsCompleted)
           {
               Thread.Sleep(200);
           }
           double minNum = double.MaxValue;
           int minIndex = 0;
           for (int i = 0; i < count; i++)
           {
               if (differ[i] < minNum)
               {
                   minNum = differ[i];
                   minIndex = i;
               }
           }
           Console.WriteLine($"The best diff: {minNum:0.####}");
           _trainedModel = transformers[minIndex];
           _mlContext.Model.Save(_trainedModel, null, _modelPath);
           _predEngine = _mlContext.Model.CreatePredictionEngine<BcwQuestion, BcwQuestionPrediction>(_trainedModel);


       }//*/



    }
}
