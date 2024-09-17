using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

class Predictions
{
    static void Main(string[] args)
    {
        string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

        MLContext mlContext = new MLContext(seed: 0);

        var models = Train(mlContext, _trainDataPath);

        Evaluate(mlContext, models, _testDataPath);

        PredictFareAndWriteToFile(mlContext, models, _trainDataPath, "train_predicted.csv");
        PredictFareAndWriteToFile(mlContext, models, _testDataPath, "test_predicted.csv");

        Console.WriteLine("Prediction completed and files saved.");
    }

    static (ITransformer fareModel, ITransformer timeModel, ITransformer consumptionModel) Train(MLContext mlContext, string dataPath)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        var farePipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded", "TripTime"))
            .Append(mlContext.Regression.Trainers.FastTree());

        var fareModel = farePipeline.Fit(dataView);

        var timePipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "TripTime")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree());

        var timeModel = timePipeline.Fit(dataView);

        var consumptionPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Consumption")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded", "TripTime"))
            .Append(mlContext.Regression.Trainers.FastTree());

        var consumptionModel = consumptionPipeline.Fit(dataView);

        return (fareModel, timeModel, consumptionModel);
    }

    static void Evaluate(MLContext mlContext, (ITransformer fareModel, ITransformer timeModel, ITransformer consumptionModel) models, string testDataPath)
    {
        IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(testDataPath, hasHeader: true, separatorChar: ',');

        var farePredictions = models.fareModel.Transform(testDataView);
        var fareMetrics = mlContext.Regression.Evaluate(farePredictions, labelColumnName: "FareAmount");

        var timePredictions = models.timeModel.Transform(testDataView);
        var timeMetrics = mlContext.Regression.Evaluate(timePredictions, labelColumnName: "TripTime");

        var consumptionPredictions = models.consumptionModel.Transform(testDataView);
        var consumptionMetrics = mlContext.Regression.Evaluate(consumptionPredictions, labelColumnName: "Consumption");

        Console.WriteLine($"*************************************************");
        Console.WriteLine($"*       Model quality metrics evaluation         ");
        Console.WriteLine($"*------------------------------------------------");
        Console.WriteLine($"*       RSquared Score (FareAmount): {fareMetrics.RSquared:0.##}");
        Console.WriteLine($"*       Root Mean Squared Error (FareAmount): {fareMetrics.RootMeanSquaredError:#.##}");
        Console.WriteLine($"*       RSquared Score (TripTime): {timeMetrics.RSquared:0.##}");
        Console.WriteLine($"*       Root Mean Squared Error (TripTime): {timeMetrics.RootMeanSquaredError:#.##}");
        Console.WriteLine($"*       RSquared Score (Consumption): {consumptionMetrics.RSquared:0.##}");
        Console.WriteLine($"*       Root Mean Squared Error (Consumption): {consumptionMetrics.RootMeanSquaredError:#.##}");
    }

    static void PredictFareAndWriteToFile(MLContext mlContext, (ITransformer fareModel, ITransformer timeModel, ITransformer consumptionModel) models, string dataPath, string outputPath)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        var farePredictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, FarePrediction>(models.fareModel);
        var timePredictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TimePrediction>(models.timeModel);
        var consumptionPredictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, ConsumptionPrediction>(models.consumptionModel);

        IEnumerable<TaxiTrip> trips = mlContext.Data.CreateEnumerable<TaxiTrip>(dataView, reuseRowObject: false);
        List<TaxiTrip> predictedTrips = new List<TaxiTrip>();

        foreach (var trip in trips)
        {
            var farePrediction = farePredictionEngine.Predict(trip);
            var timePrediction = timePredictionEngine.Predict(trip);
            var consumptionPrediction = consumptionPredictionEngine.Predict(trip);

            trip.PredictedFareAmount = farePrediction.FareAmount;
            trip.PredictedTripTime = timePrediction.TripTime;
            trip.PredictedConsumption = consumptionPrediction.Consumption;  
            predictedTrips.Add(trip);
        }

        using (var writer = new StreamWriter(outputPath))
        {
            writer.WriteLine("VendorId,RateCode,PassengerCount,TripTime,TripDistance,PaymentType,FareAmount,Consumption,PredictedFareAmount,PredictedTripTime,PredictedConsumption");  // Update header

            foreach (var trip in predictedTrips)
            {
                writer.WriteLine($"{trip.VendorId},{trip.RateCode},{trip.PassengerCount},{trip.TripTime},{trip.TripDistance},{trip.PaymentType},{trip.FareAmount},{trip.Consumption},{trip.PredictedFareAmount},{trip.PredictedTripTime},{trip.PredictedConsumption}");  // Update row
            }
        }
    }
}