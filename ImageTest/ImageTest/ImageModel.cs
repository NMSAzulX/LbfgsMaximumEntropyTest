using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageLearning
{

    public class ImageInput
    {
        public byte[] Image { get; set; }

        public UInt32 LabelAsKey { get; set; }

        public string ImagePath { get; set; }

        public string Label { get; set; }
    }


    struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }

    public class ImagePrediction : ImageModel
    {
        public float[] Score { get; set; }

        public string PredictedLabelValue { get; set; }
    }

    public class ImageModel
    {
        public ImageModel()
        {

        }
        public ImageModel(string imagePath, string label)
        {
            ImagePath = imagePath;
            Label = label;
        }

        public string ImagePath { get; set; }

        public string Label { get; set; }
    }

    public class ImageOutput
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }
}
