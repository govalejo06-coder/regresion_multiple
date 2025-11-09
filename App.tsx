import React, { useState, useCallback, useMemo, useEffect } from 'react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { mean, median, standardDeviation, min, max, sampleCorrelation } from 'simple-statistics';
import { MultivariateLinearRegression } from 'ml-regression';
import Markdown from 'react-markdown';

import { getDatasetInsights, getVariableSuggestions } from './services/geminiService';
import type { DataRow, DataSet, DescriptiveStats, CorrelationMatrix, ModelResults, TrainedModel } from './types';
import { 
    UploadCloudIcon, BarChartIcon, BrainCircuitIcon, TargetIcon,
    ClipboardListIcon, FileCheckIcon, FilterXIcon, SlidersHorizontalIcon,
    GitForkIcon, ClipboardCheckIcon, ShieldCheckIcon, LightbulbIcon,
    CheckCircle2Icon, CircleDotIcon, CircleIcon
} from './components/icons';

const Section: React.FC<{ title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => (
    <div className="bg-slate-800/50 rounded-xl shadow-lg border border-slate-700 overflow-hidden mb-8 animate-fade-in">
        <div className="flex items-center p-4 bg-slate-800 border-b border-slate-700">
            <div className="text-cyan-400 mr-3">{icon}</div>
            <h2 className="text-xl font-bold text-slate-100">{title}</h2>
        </div>
        <div className="p-6">
            {children}
        </div>
    </div>
);

const FileUpload: React.FC<{ onFileParsed: (dataSet: DataSet) => void; setIsLoading: (loading: boolean) => void }> = ({ onFileParsed, setIsLoading }) => {
    const [dragActive, setDragActive] = useState(false);

    const handleFile = useCallback((file: File) => {
        setIsLoading(true);
        if (file.name.endsWith('.csv')) {
            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    processParsedData(results.data as DataRow[]);
                },
                error: (err) => {
                    alert('Error al parsear CSV: ' + err.message);
                    setIsLoading(false);
                }
            });
        } else if (file.name.endsWith('.xlsx')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = new Uint8Array(e.target?.result as ArrayBuffer);
                    const workbook = XLSX.read(data, { type: 'array' });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];
                    const jsonData = XLSX.utils.sheet_to_json(worksheet) as DataRow[];
                    processParsedData(jsonData);
                } catch (err) {
                    alert('Error al leer el archivo Excel.');
                    setIsLoading(false);
                }
            };
            reader.readAsArrayBuffer(file);
        } else {
            alert('Formato de archivo no soportado. Por favor, sube un .csv o .xlsx');
            setIsLoading(false);
        }
    }, [setIsLoading]);

    const processParsedData = (data: DataRow[]) => {
        if (data.length === 0) {
            alert("El archivo está vacío o no tiene datos.");
            setIsLoading(false);
            return;
        }
        const headers = Object.keys(data[0]);
        const numericHeaders: string[] = [];
        const processedData = data.map(row => {
            const newRow: DataRow = {};
            for (const key of headers) {
                const value = parseFloat(String(row[key]));
                if (!isNaN(value)) {
                    newRow[key] = value;
                    if (!numericHeaders.includes(key)) {
                        numericHeaders.push(key);
                    }
                } else {
                    newRow[key] = String(row[key]);
                }
            }
            return newRow;
        });

        onFileParsed({ data: processedData, headers, numericHeaders });
        setIsLoading(false);
    };
    
    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };
    
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    return (
        <form id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()} className="relative w-full">
            <input type="file" id="input-file-upload" accept=".csv, .xlsx" className="hidden" onChange={handleChange} />
            <label 
                htmlFor="input-file-upload" 
                className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-300
                    ${dragActive ? 'border-cyan-400 bg-slate-700' : 'border-slate-600 hover:border-slate-500 bg-slate-800 hover:bg-slate-700/50'}`}
            >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <UploadCloudIcon className="w-10 h-10 mb-3 text-slate-400" />
                    <p className="mb-2 text-sm text-slate-400">
                        <span className="font-semibold text-cyan-400">Haz clic para subir</span> o arrastra y suelta
                    </p>
                    <p className="text-xs text-slate-500">CSV o XLSX</p>
                </div>
            </label>
            {dragActive && <div className="absolute w-full h-full top-0 left-0" onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}></div>}
        </form>
    );
};

interface AnalysisDisplayProps {
  dataSet: DataSet;
  onVariableSuggestions: (suggestions: { dependentVar: string; independentVars: string[] }) => void;
}

const AnalysisDisplay: React.FC<AnalysisDisplayProps> = ({ dataSet, onVariableSuggestions }) => {
    const { data, numericHeaders } = dataSet;
    const [insights, setInsights] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSuggesting, setIsSuggesting] = useState(false);
    const [suggestionError, setSuggestionError] = useState<string | null>(null);

    const stats = useMemo<DescriptiveStats>(() => {
        const descStats: DescriptiveStats = {};
        numericHeaders.forEach(header => {
            const values = data.map(row => row[header] as number).filter(v => v !== null && !isNaN(v));
            if (values.length > 0) {
                descStats[header] = {
                    count: values.length,
                    min: min(values),
                    max: max(values),
                    mean: mean(values),
                    median: median(values),
                    std: standardDeviation(values),
                };
            }
        });
        return descStats;
    }, [data, numericHeaders]);

    const correlationMatrix = useMemo<CorrelationMatrix>(() => {
        const matrix: CorrelationMatrix = {};
        numericHeaders.forEach(h1 => {
            matrix[h1] = {};
            numericHeaders.forEach(h2 => {
                if (h1 === h2) {
                    matrix[h1][h2] = 1;
                } else if (matrix[h2] && typeof matrix[h2][h1] !== 'undefined') {
                    matrix[h1][h2] = matrix[h2][h1];
                } else {
                    const v1 = data.map(row => row[h1] as number);
                    const v2 = data.map(row => row[h2] as number);
                    matrix[h1][h2] = sampleCorrelation(v1, v2);
                }
            });
        });
        return matrix;
    }, [data, numericHeaders]);

    const handleGenerateInsights = async () => {
        setIsLoading(true);
        const analysis = await getDatasetInsights(dataSet.headers, stats, data.slice(0, 5));
        setInsights(analysis);
        setIsLoading(false);
    };

    const handleSuggestVariables = async () => {
        setIsSuggesting(true);
        setSuggestionError(null);
        try {
            const suggestions = await getVariableSuggestions(dataSet.headers, data);
            onVariableSuggestions(suggestions);
        } catch(error) {
            setSuggestionError(error instanceof Error ? error.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsSuggesting(false);
        }
    };

    const getCorrelationColor = (value: number) => {
        if (isNaN(value)) return 'bg-slate-700';
        const alpha = Math.abs(value);
        if (value > 0) return `rgba(56, 189, 248, ${alpha})`; // sky-500
        return `rgba(251, 113, 133, ${alpha})`; // rose-500
    };

    return (
        <>
            <h3 className="text-lg font-semibold mb-4 text-slate-300">Resumen del Dataset</h3>
            <p className="mb-4 text-slate-400">Total de registros: {data.length}, Total de columnas: {dataSet.headers.length}</p>
            
            <h3 className="text-lg font-semibold mt-6 mb-4 text-slate-300">Estadísticas Descriptivas</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-slate-400">
                    <thead className="text-xs text-slate-300 uppercase bg-slate-700">
                        <tr>
                            <th scope="col" className="px-6 py-3">Métrica</th>
                            {numericHeaders.map(h => <th key={h} scope="col" className="px-6 py-3">{h}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {Object.keys(stats[numericHeaders[0]] || {}).map(metric => (
                            <tr key={metric} className="bg-slate-800 border-b border-slate-700 hover:bg-slate-700/50">
                                <td className="px-6 py-4 font-medium text-slate-200 whitespace-nowrap">{metric}</td>
                                {numericHeaders.map(h => <td key={`${h}-${metric}`} className="px-6 py-4">{stats[h]?.[metric]?.toFixed(2) ?? 'N/A'}</td>)}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <h3 className="text-lg font-semibold mt-6 mb-4 text-slate-300">Matriz de Correlación</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-slate-400 border-collapse">
                    <thead>
                        <tr>
                            <th className="p-2 border border-slate-600 bg-slate-700"></th>
                            {numericHeaders.map(h => <th key={h} className="p-2 border border-slate-600 bg-slate-700 text-slate-300">{h}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {numericHeaders.map(h1 => (
                            <tr key={h1}>
                                <th className="p-2 border border-slate-600 bg-slate-700 text-slate-300">{h1}</th>
                                {numericHeaders.map(h2 => (
                                    <td key={`${h1}-${h2}`} className="p-2 border border-slate-600 text-center font-mono text-white" style={{ backgroundColor: getCorrelationColor(correlationMatrix[h1][h2])}}>
                                        {correlationMatrix[h1][h2].toFixed(2)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <h3 className="text-lg font-semibold mt-6 mb-4 text-slate-300">Análisis con IA de Gemini</h3>
            <button 
                onClick={handleGenerateInsights}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white font-semibold rounded-lg shadow-md transition-colors duration-300 disabled:bg-slate-600 disabled:cursor-not-allowed">
                {isLoading ? 'Generando...' : 'Generar Análisis'}
            </button>
            {isLoading && <div className="mt-4 text-center">Cargando análisis...</div>}
            {insights && (
                <div className="mt-4 p-4 bg-slate-900/50 rounded-lg border border-slate-700 prose prose-invert prose-sm max-w-none">
                    <Markdown>{insights}</Markdown>
                </div>
            )}
            
            <h3 className="text-lg font-semibold mt-8 mb-4 text-slate-300">Asistente de Modelo con IA</h3>
            <p className="text-slate-400 mb-4">
                Deja que Gemini analice tus columnas y sugiera automáticamente las variables dependientes e independientes para tu modelo de regresión.
            </p>
            <button 
                onClick={handleSuggestVariables}
                disabled={isSuggesting}
                className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-md transition-colors duration-300 disabled:bg-slate-600 disabled:cursor-not-allowed">
                {isSuggesting ? 'Sugiriendo...' : 'Sugerir Variables con IA'}
            </button>
            {suggestionError && <p className="text-rose-400 mt-2 text-sm">{suggestionError}</p>}
        </>
    );
};

interface ModelTrainerProps {
    dataSet: DataSet;
    onModelTrain: (model: TrainedModel, results: ModelResults, independentVars: string[], dependentVar: string) => void;
    suggestions: { dependentVar: string; independentVars: string[] } | null;
}

const ModelTrainer: React.FC<ModelTrainerProps> = ({ dataSet, onModelTrain, suggestions }) => {
    const [dependentVar, setDependentVar] = useState<string>('');
    const [independentVars, setIndependentVars] = useState<string[]>([]);

    useEffect(() => {
        if (suggestions && dataSet.headers.includes(suggestions.dependentVar)) {
            if (dataSet.numericHeaders.includes(suggestions.dependentVar)) {
                setDependentVar(suggestions.dependentVar);
                const validIndependent = suggestions.independentVars.filter(v => 
                    dataSet.numericHeaders.includes(v) && v !== suggestions.dependentVar
                );
                setIndependentVars(validIndependent);
            }
        }
    }, [suggestions, dataSet.headers, dataSet.numericHeaders]);

    const handleTrainModel = () => {
        if (!dependentVar || independentVars.length === 0) {
            alert('Por favor, selecciona la variable dependiente y al menos una independiente.');
            return;
        }

        const allVars = [dependentVar, ...independentVars];
        const cleanData = dataSet.data.filter(row =>
            allVars.every(v => typeof row[v] === 'number' && !isNaN(row[v] as number))
        );

        if (cleanData.length < independentVars.length + 2) {
            alert('No hay suficientes datos limpios (numéricos y sin valores faltantes) para entrenar el modelo con las variables seleccionadas. Por favor, verifica tu archivo de datos.');
            return;
        }

        const y = cleanData.map(row => row[dependentVar] as number);
        const x = cleanData.map(row => independentVars.map(key => row[key] as number));
        
        try {
            const model = new MultivariateLinearRegression(x, y);

            if (model.weights.some(w => isNaN(w[0]))) {
                 alert("Error al entrenar el modelo. Los coeficientes resultantes no son válidos (NaN). Esto puede deberse a una multicolinealidad perfecta (variables predictoras idénticas o muy correlacionadas).");
                 return;
            }
            
            const predictions = x.map(row => model.predict(row));
            const yMean = mean(y);
            const ssTotal = y.reduce((acc, val) => acc + Math.pow(val - yMean, 2), 0);

            if (ssTotal === 0) {
                alert("No se puede entrenar el modelo: la variable dependiente tiene varianza cero (todos los valores son iguales).");
                return;
            }
            
            const ssResidual = y.reduce((acc, val, i) => acc + Math.pow(val - predictions[i], 2), 0);
            
            const n = cleanData.length;
            const k = independentVars.length;
            
            const rSquared = 1 - (ssResidual / ssTotal);
            const rSquaredAdjusted = 1 - ((1 - rSquared) * (n - 1)) / (n - k - 1);
            const rmse = Math.sqrt(ssResidual / n);

            const results: ModelResults = {
                coefficients: model.weights.slice(1).flat(),
                intercept: model.weights[0][0],
                rSquared,
                rSquaredAdjusted,
                rmse,
            };
            onModelTrain(model, results, independentVars, dependentVar);
        } catch (error) {
            alert("Ocurrió un error inesperado al entrenar el modelo. Revisa la consola para más detalles.");
            console.error(error);
        }
    };
    
    const handleIndependentVarToggle = (varName: string) => {
        setIndependentVars(prev => 
            prev.includes(varName) ? prev.filter(v => v !== varName) : [...prev, varName]
        );
    };
    
    const availableIndependentVars = dataSet.numericHeaders.filter(h => h !== dependentVar);

    return (
        <>
            <div className="grid md:grid-cols-2 gap-6 mb-6">
                <div>
                    <label htmlFor="dependent-var" className="block mb-2 text-sm font-medium text-slate-300">1. Selecciona la Variable Dependiente (Y)</label>
                    <select id="dependent-var" value={dependentVar} onChange={e => {setDependentVar(e.target.value); setIndependentVars([])}} className="bg-slate-700 border border-slate-600 text-white text-sm rounded-lg focus:ring-cyan-500 focus:border-cyan-500 block w-full p-2.5">
                        <option value="">-- Ventas, Ingresos, etc. --</option>
                        {dataSet.numericHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                    </select>
                </div>
                <div>
                    <h3 className="mb-2 text-sm font-medium text-slate-300">2. Selecciona las Variables Independientes (X)</h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-32 overflow-y-auto p-2 bg-slate-700/50 rounded-lg">
                        {availableIndependentVars.map(h => (
                             <label key={h} className="flex items-center space-x-2 text-sm cursor-pointer p-1 rounded hover:bg-slate-600">
                                <input type="checkbox" checked={independentVars.includes(h)} onChange={() => handleIndependentVarToggle(h)} className="w-4 h-4 text-cyan-600 bg-slate-600 border-slate-500 rounded focus:ring-cyan-500 focus:ring-2" />
                                <span>{h}</span>
                            </label>
                        ))}
                    </div>
                </div>
            </div>
            <button onClick={handleTrainModel} disabled={!dependentVar || independentVars.length === 0} className="w-full px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white font-bold rounded-lg shadow-md transition-colors duration-300 disabled:bg-slate-600 disabled:cursor-not-allowed">
                Entrenar Modelo de Regresión
            </button>
        </>
    );
};

const Predictor: React.FC<{ model: TrainedModel; results: ModelResults; independentVars: string[]; dependentVar: string; }> = ({ model, results, independentVars, dependentVar }) => {
    const [inputs, setInputs] = useState<Record<string, number>>(
        independentVars.reduce((acc, v) => ({...acc, [v]: 0}), {})
    );
    const [prediction, setPrediction] = useState<number | null>(null);

    const handleInputChange = (varName: string, value: string) => {
        setInputs(prev => ({ ...prev, [varName]: parseFloat(value) || 0 }));
    };

    const handlePredict = () => {
        const inputValues = independentVars.map(v => inputs[v]);
        if (inputValues.some(isNaN)) {
            alert('Por favor, ingresa valores numéricos válidos.');
            return;
        }
        const pred = model.predict(inputValues);
        setPrediction(pred);
    };

    return (
        <>
        <div className="grid md:grid-cols-2 gap-8">
            <div>
                <h3 className="text-lg font-semibold mb-4 text-slate-300">Resultados del Modelo</h3>
                <div className="space-y-3">
                    <div className="flex justify-between p-3 bg-slate-700/50 rounded-lg"><span>R² (R-squared):</span> <span className="font-mono text-cyan-400">{results.rSquared.toFixed(4)}</span></div>
                    <div className="flex justify-between p-3 bg-slate-700/50 rounded-lg"><span>R² Ajustado:</span> <span className="font-mono text-cyan-400">{results.rSquaredAdjusted.toFixed(4)}</span></div>
                    <div className="flex justify-between p-3 bg-slate-700/50 rounded-lg"><span>RMSE:</span> <span className="font-mono text-cyan-400">{results.rmse.toFixed(4)}</span></div>
                    <div className="flex justify-between p-3 bg-slate-700/50 rounded-lg"><span>Intercepto:</span> <span className="font-mono text-cyan-400">{results.intercept.toFixed(4)}</span></div>
                </div>
                <h4 className="text-md font-semibold mt-6 mb-2 text-slate-300">Coeficientes</h4>
                <div className="space-y-2">
                {independentVars.map((v, i) => (
                    <div key={v} className="flex justify-between p-2 bg-slate-700/50 rounded"><span>{v}:</span> <span className="font-mono text-sky-400">{results.coefficients[i].toFixed(4)}</span></div>
                ))}
                </div>
            </div>
            <div>
                 <h3 className="text-lg font-semibold mb-4 text-slate-300">Realizar una Predicción</h3>
                 <div className="space-y-4 mb-4">
                    {independentVars.map(v => (
                        <div key={v}>
                            <label htmlFor={`pred-${v}`} className="block mb-1 text-sm font-medium text-slate-400">{v}</label>
                            <input
                                type="number"
                                id={`pred-${v}`}
                                value={inputs[v]}
                                onChange={(e) => handleInputChange(v, e.target.value)}
                                className="bg-slate-700 border border-slate-600 text-white text-sm rounded-lg focus:ring-cyan-500 focus:border-cyan-500 block w-full p-2.5"
                            />
                        </div>
                    ))}
                 </div>
                 <button onClick={handlePredict} className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg shadow-md transition-colors duration-300">
                    Predecir
                </button>
                {prediction !== null && (
                    <div className="mt-6 text-center p-6 bg-slate-900/50 rounded-xl border border-green-500">
                        <p className="text-lg text-slate-300">Predicción de <span className="font-bold text-green-400">{dependentVar}</span>:</p>
                        <p className="text-4xl font-bold text-white mt-2">{prediction.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                    </div>
                )}
            </div>
        </div>
        </>
    );
};

const workflowSteps = [
    { id: 1, title: 'Definición del problema', icon: <ClipboardListIcon className="w-5 h-5" />, section: 'start' },
    { id: 2, title: 'Carga y revisión de datos', icon: <FileCheckIcon className="w-5 h-5" />, section: 'upload' },
    { id: 3, title: 'Análisis exploratorio (EDA)', icon: <BarChartIcon className="w-5 h-5" />, section: 'analysis' },
    { id: 4, title: 'Detección de outliers', icon: <FilterXIcon className="w-5 h-5" />, section: 'analysis' },
    { id: 5, title: 'Ingeniería de variables', icon: <SlidersHorizontalIcon className="w-5 h-5" />, section: 'analysis' },
    { id: 6, title: 'División del dataset', icon: <GitForkIcon className="w-5 h-5" />, section: 'train' },
    { id: 7, title: 'Entrenamiento del modelo', icon: <BrainCircuitIcon className="w-5 h-5" />, section: 'train' },
    { id: 8, title: 'Evaluación del modelo', icon: <ClipboardCheckIcon className="w-5 h-5" />, section: 'predict' },
    { id: 9, title: 'Diagnóstico de supuestos', icon: <ShieldCheckIcon className="w-5 h-5" />, section: 'predict' },
    { id: 10, title: 'Interpretación de resultados', icon: <LightbulbIcon className="w-5 h-5" />, section: 'predict' },
    { id: 11, title: 'Predicción', icon: <TargetIcon className="w-5 h-5" />, section: 'predict' },
];

const WorkflowSidebar: React.FC<{ dataSet: DataSet | null, model: any | null }> = ({ dataSet, model }) => {
    const isDataLoaded = !!dataSet;
    const isModelTrained = !!model;

    const getStepStatus = (stepId: number) => {
        if (!isDataLoaded) {
            return stepId === 1 ? 'active' : 'pending';
        }
        if (!isModelTrained) {
            if (stepId <= 2) return 'completed';
            if (stepId > 2 && stepId <= 5) return 'active';
            return 'pending';
        }
        if (isModelTrained) {
            if (stepId <= 7) return 'completed';
            if (stepId > 7) return 'active';
        }
        return 'pending';
    };

    const StatusIcon: React.FC<{ status: 'completed' | 'active' | 'pending' }> = ({ status }) => {
        if (status === 'completed') return <CheckCircle2Icon className="w-6 h-6 text-green-400" />;
        if (status === 'active') return <CircleDotIcon className="w-6 h-6 text-cyan-400 animate-pulse" />;
        return <CircleIcon className="w-6 h-6 text-slate-600" />;
    };

    return (
        <aside className="md:col-span-4 lg:col-span-3">
            <div className="sticky top-8 bg-slate-800/50 rounded-xl shadow-lg border border-slate-700 p-6">
                <h3 className="text-lg font-bold text-slate-100 mb-4">Guía de Flujo de Trabajo</h3>
                <ol className="relative border-l border-slate-700">
                    {workflowSteps.map(step => {
                        const status = getStepStatus(step.id);
                        const isCompleted = status === 'completed';
                        const isActive = status === 'active';
                        
                        return (
                            <li key={step.id} className="mb-6 ml-6">
                                <span className="absolute flex items-center justify-center w-6 h-6 bg-slate-800 rounded-full -left-3 ring-8 ring-slate-800">
                                    <StatusIcon status={status} />
                                </span>
                                <div className={`flex items-center transition-opacity duration-300 ${isCompleted || isActive ? 'opacity-100' : 'opacity-50'}`}>
                                    <div className={`mr-2 ${isActive ? 'text-cyan-300' : isCompleted ? 'text-green-400' : 'text-slate-500'}`}>{step.icon}</div>
                                    <h4 className={`text-md font-semibold ${isActive ? 'text-slate-100' : 'text-slate-300'}`}>{step.title}</h4>
                                </div>
                            </li>
                        );
                    })}
                </ol>
            </div>
        </aside>
    );
};

export default function App() {
    const [isLoading, setIsLoading] = useState(false);
    const [dataSet, setDataSet] = useState<DataSet | null>(null);
    const [model, setModel] = useState<{ model: TrainedModel; results: ModelResults; independentVars: string[], dependentVar: string } | null>(null);
    const [variableSuggestions, setVariableSuggestions] = useState<{ dependentVar: string; independentVars: string[] } | null>(null);

    const handleFileParsed = useCallback((newDataSet: DataSet) => {
        setDataSet(newDataSet);
        setModel(null);
        setVariableSuggestions(null);
    }, []);

    const handleModelTrain = useCallback((model: TrainedModel, results: ModelResults, independentVars: string[], dependentVar: string) => {
        setModel({ model, results, independentVars, dependentVar });
    }, []);
    
    return (
        <div className="container mx-auto p-4 md:p-8">
            <header className="text-center mb-10">
                <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-cyan-300">
                    Predicción de Ventas con IA
                </h1>
                <p className="mt-2 text-lg text-slate-400">
                    Sigue la guía, carga tus datos, entrena un modelo y predice el futuro.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
                <WorkflowSidebar dataSet={dataSet} model={model} />

                <main className="md:col-span-8 lg:col-span-9">
                    <Section title="1. Cargar Datos" icon={<UploadCloudIcon />}>
                        <p className="text-slate-400 mb-4">Sube tu archivo de datos en formato CSV o Excel. Asegúrate de que la primera fila contenga los nombres de las columnas.</p>
                        {isLoading ? 
                            <div className="flex justify-center items-center h-64"><div className="animate-spin rounded-full h-16 w-16 border-b-2 border-cyan-400"></div></div> : 
                            <FileUpload onFileParsed={handleFileParsed} setIsLoading={setIsLoading} />
                        }
                    </Section>
                    
                    {dataSet && (
                        <Section title="2. Análisis del Dataset" icon={<BarChartIcon />}>
                            <AnalysisDisplay dataSet={dataSet} onVariableSuggestions={setVariableSuggestions} />
                        </Section>
                    )}
                    
                    {dataSet && (
                        <Section title="3. Entrenar Modelo" icon={<BrainCircuitIcon />}>
                            <ModelTrainer dataSet={dataSet} onModelTrain={handleModelTrain} suggestions={variableSuggestions} />
                        </Section>
                    )}

                    {model && (
                        <Section title="4. Predecir Ventas" icon={<TargetIcon />}>
                            <Predictor 
                                model={model.model} 
                                results={model.results} 
                                independentVars={model.independentVars} 
                                dependentVar={model.dependentVar}
                            />
                        </Section>
                    )}
                </main>
            </div>
        </div>
    );
}