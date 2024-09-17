using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

public class TaxiTrip
{
    [LoadColumn(0)]
    public string? VendorId;

    [LoadColumn(1)]
    public string? RateCode;

    [LoadColumn(2)]
    public float PassengerCount;

    [LoadColumn(3)]
    public float TripTime;

    [LoadColumn(4)]
    public float TripDistance;

    [LoadColumn(5)]
    public string? PaymentType;

    [LoadColumn(6)]
    public float FareAmount;

    [LoadColumn(7)]
    public float Consumption;  

    [LoadColumn(8)]
    public float PredictedFareAmount { get; set; }

    [LoadColumn(9)]
    public float PredictedTripTime { get; set; }

    [LoadColumn(10)]
    public float PredictedConsumption { get; set; } 
}

public class FarePrediction
{
    [ColumnName("Score")]
    public float FareAmount;
}

public class TimePrediction
{
    [ColumnName("Score")]
    public float TripTime;
}

public class ConsumptionPrediction
{
    [ColumnName("Score")]
    public float Consumption;
}