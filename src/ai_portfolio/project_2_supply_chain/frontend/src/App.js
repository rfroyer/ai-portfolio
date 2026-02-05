import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
    const [products, setProducts] = useState([]);
    const [selectedProductId, setSelectedProductId] = useState(null);
    const [forecastData, setForecastData] = useState(null);
    const [productSummary, setProductSummary] = useState(null);
    const [isSummaryLoading, setIsSummaryLoading] = useState(true);
    const [isChartLoading, setIsChartLoading] = useState(true);
    const [chartKey, setChartKey] = useState(0);

    useEffect(() => {
        axios.get('http://localhost:8000/products')
            .then(response => {
                const productList = response.data.products || [];
                const uniqueProducts = Array.from(new Map(productList.map(p => [p.product_id, p])).values());
                setProducts(uniqueProducts);
                if (uniqueProducts.length > 0) {
                    setSelectedProductId(uniqueProducts[0].product_id.toString());
                }
            })
            .catch(error => console.error("FATAL: Could not load products!", error));
    }, []);

    useEffect(() => {
        if (!selectedProductId) return;

        let isMounted = true;

        // Fetch Summary
        setIsSummaryLoading(true);
        axios.get(`http://localhost:8000/product-summary/${selectedProductId}`)
            .then(response => {
                if (isMounted) setProductSummary(response.data);
            })
            .catch(error => console.error(`Summary error: ${error}`))
            .finally(() => {
                if (isMounted) setIsSummaryLoading(false);
            });

        // Fetch Forecast
        setIsChartLoading(true);
        axios.get(`http://localhost:8000/forecast/${selectedProductId}`)
            .then(response => {
                if (isMounted) {
                    const forecast = response.data.forecast;
                    if (forecast && forecast.length > 0) {
                        setForecastData({
                            labels: forecast.map(d => new Date(d.ds).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
                            datasets: [{
                                label: 'Forecasted Demand',
                                data: forecast.map(d => d.yhat),
                                borderColor: 'rgb(75, 192, 192)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                fill: true,
                                tension: 0.2
                            }]
                        });
                        setChartKey(prevKey => prevKey + 1);
                    } else {
                        setForecastData(null);
                    }
                }
            })
            .catch(error => {
                console.error(`Forecast error: ${error}`);
                if (isMounted) setForecastData(null);
            })
            .finally(() => {
                if (isMounted) setIsChartLoading(false);
            });

        return () => { isMounted = false; };
    }, [selectedProductId]);

    return (
        <div className="App">
            <header className="header"><h1>📊 Supply Chain Optimization Dashboard</h1></header>
            <div className="container">
                <div className="summary-cards">
                    <div className="card">
                        <h3>TOTAL REVENUE</h3>
                        <p className="value">{isSummaryLoading ? '...' : `$${productSummary?.total_revenue?.toLocaleString('en-US', { maximumFractionDigits: 2 }) || '0'}`}</p>
                    </div>
                    <div className="card">
                        <h3>TOTAL TRANSACTIONS</h3>
                        <p className="value">{isSummaryLoading ? '...' : productSummary?.total_transactions?.toLocaleString() || '0'}</p>
                    </div>
                    <div className="card">
                        <h3>CURRENT INVENTORY</h3>
                        {/* --- THE COSMETIC FIX IS HERE --- */}
                        <p className="value">{isSummaryLoading ? '...' : productSummary?.total_inventory?.toLocaleString() || '0'}</p>
                    </div>
                </div>
                <div className="product-selector">
                    <label htmlFor="product-select">Select Product:</label>
                    <select id="product-select" value={selectedProductId || ''} onChange={e => setSelectedProductId(e.target.value)}>
                        {products.map(product => (
                            <option key={product.product_id} value={product.product_id}>{product.product_name} (ID: {product.product_id})</option>
                        ))}
                    </select>
                </div>
                <div className="forecast-section">
                    <h2>30-Day Demand Forecast</h2>
                    <div className="chart-container">
                        {isChartLoading && <p>🧠 Finding the best forecast model...</p>}
                        {!isChartLoading && forecastData && (
                            <Line key={chartKey} data={forecastData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: false } } }} />
                        )}
                        {!isChartLoading && !forecastData && <p>No forecast data available for this product.</p>}
                    </div>
                </div>
            </div>
            <footer className="footer"><p>Supply Chain Optimization System © 2024</p></footer>
        </div>
    );
}

export default App;
