import { MultivariateLinearRegression } from 'ml-regression';

export type DataRow = Record<string, string | number>;

export interface DataSet {
  data: DataRow[];
  headers: string[];
  numericHeaders: string[];
}

export interface DescriptiveStats {
  [key: string]: {
    count: number;
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  };
}

export interface CorrelationMatrix {
  [key: string]: {
    [key: string]: number;
  };
}

export interface ModelResults {
  coefficients: number[];
  intercept: number;
  rSquared: number;
  rSquaredAdjusted: number;
  rmse: number;
}

export type TrainedModel = MultivariateLinearRegression;
