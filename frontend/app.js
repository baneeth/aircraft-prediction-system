 /*
  ================================================================================
  AIRCRAFT PREDICTION SYSTEM - JAVASCRIPT APPLICATION
  ================================================================================
  Handles API communication, form submission, and UI interactions
  ================================================================================
  */

  // API Configuration
  const API_BASE_URL = 'https://aircraft-prediction-system.onrender.com';

  // Initialize application when DOM is loaded
  document.addEventListener('DOMContentLoaded', function() {
      initializeApp();
  });

  /**
   * Initialize Application
   */
  function initializeApp() {
      console.log('Aircraft Prediction System - Initializing...');

      // Setup event listeners
      setupTabNavigation();
      setupFormSubmission();

      // Load initial data
      checkAPIStatus();
      loadModelInfo();

      console.log('Application initialized successfully');
  }
  /**
   * Tab Navigation
   */
  function setupTabNavigation() {
      const tabs = document.querySelectorAll('.nav-tab');
      const tabContents = document.querySelectorAll('.tab-content');

      tabs.forEach(tab => {
          tab.addEventListener('click', function() {
              const targetTab = this.getAttribute('data-tab');

              // Remove active class from all tabs and contents
              tabs.forEach(t => t.classList.remove('active'));
              tabContents.forEach(content => content.classList.remove('active'));

              // Add active class to clicked tab and corresponding content
              this.classList.add('active');
              document.getElementById(`${targetTab}-tab`).classList.add('active');

              // Load analytics when Analytics tab is clicked
              if (targetTab === 'analytics') {
                  const analyticsContent = document.getElementById('analytics-content');
                  // Only load once
                  if (!analyticsContent.dataset.loaded) {
                      console.log('Loading analytics data...');
                      loadAnalytics();
                      analyticsContent.dataset.loaded = 'true';
                  }
              }

              // Setup batch processing when Batch tab is clicked
              if (targetTab === 'batch' && !batchProcessingInitialized) {
                  console.log('Setting up batch processing...');
                  setupBatchProcessing();
              }
          });
      });
  }

  /**
   * Check API Status
   */
  async function checkAPIStatus() {
      try {
          const response = await fetch(`${API_BASE_URL}/`);
          const data = await response.json();

          const statusElement = document.getElementById('api-status');

          if (data.status === 'online' && data.models_loaded) {
              statusElement.innerHTML = '<span class="status-dot"></span> Online';
              statusElement.style.color = 'var(--success-green)';
              document.getElementById('models-status').textContent = '2/2';
          } else {
              statusElement.innerHTML = '<span class="status-dot"></span> Offline';
              statusElement.style.color = 'var(--danger-red)';
              document.getElementById('models-status').textContent = '0/2';
          }
      } catch (error) {
          console.error('API Status Check Failed:', error);
          const statusElement = document.getElementById('api-status');
          statusElement.innerHTML = '<span class="status-dot"></span> Offline';
          statusElement.style.color = 'var(--danger-red)';
          document.getElementById('models-status').textContent = '0/2';
      }
  }

  /**
   * Load Model Information
   */
  async function loadModelInfo() {
      try {
          const response = await fetch(`${API_BASE_URL}/api/models/info`);
          const data = await response.json();

          const infoGrid = document.getElementById('model-info');
          infoGrid.innerHTML = `
              <div class="info-item">
                  <span class="info-label">Equipment Failure Model</span>
                  <span class="info-value">${data.equipment_failure_model}</span>
              </div>
              <div class="info-item">
                  <span class="info-label">Cancellation Model</span>
                  <span class="info-value">${data.flight_cancellation_model}</span>
              </div>
              <div class="info-item">
                  <span class="info-label">Preprocessing</span>
                  <span class="info-value">${data.preprocessing_pipeline}</span>
              </div>
          `;
      } catch (error) {
          console.error('Failed to load model info:', error);
          document.getElementById('model-info').innerHTML = `
              <div class="info-item">
                  <span class="info-label">Status</span>
                  <span class="info-value" style="color: var(--danger-red);">Failed to load</span>
              </div>
          `;
      }
  }

  /**
   * Setup Form Submission
   */
  function setupFormSubmission() {
      const form = document.getElementById('prediction-form');

      form.addEventListener('submit', async function(e) {
          e.preventDefault();

          // Show loading state
          const submitBtn = document.getElementById('predict-btn');
          const btnText = submitBtn.querySelector('.btn-text');
          const btnLoader = submitBtn.querySelector('.btn-loader');

          btnText.textContent = 'Analyzing...';
          btnLoader.style.display = 'inline-block';
          submitBtn.disabled = true;

          // Trigger airplane takeoff animation
          triggerAirplaneTakeoff();

          // Collect form data
          const formData = new FormData(form);
          const flightData = {};

          formData.forEach((value, key) => {
              // Convert numeric fields to numbers
              if (['aircraft_age', 'operational_hours', 'days_since_maintenance',
                   'maintenance_count', 'distance', 'flight_duration', 'weather_severity',
                   'engine_temperature', 'oil_pressure', 'vibration_level',
                   'fuel_consumption', 'hydraulic_pressure', 'cabin_pressure',
                   'previous_delays', 'crew_experience'].includes(key)) {
                  flightData[key] = parseFloat(value);
              } else {
                  flightData[key] = value;
              }
          });

          try {
              // Make API request
              const response = await fetch(`${API_BASE_URL}/api/predict`, {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify(flightData)
              });

              const result = await response.json();

              if (result.success) {
                  displayResults(result);
              } else {
                  alert('Prediction failed: ' + result.error);
              }

          } catch (error) {
              console.error('Prediction Error:', error);
              alert('Failed to get prediction. Make sure the API server is running.');
          } finally {
              // Reset button state
              btnText.textContent = 'Analyze Flight Risk';
              btnLoader.style.display = 'none';
              submitBtn.disabled = false;
          }
      });
  }

  /**
   * Display Prediction Results
   */
  function displayResults(result) {
      const resultsCard = document.getElementById('results-card');

      // Show results card with animation
      resultsCard.style.display = 'block';

      // Update timestamp
      const timestamp = new Date(result.timestamp).toLocaleString();
      document.getElementById('results-timestamp').textContent = `Generated: ${timestamp}`;

      // Equipment Failure Results
      const equipmentPred = result.predictions.equipment_failure;
      document.getElementById('equipment-prediction').textContent = equipmentPred.prediction;
      document.getElementById('equipment-probability').textContent = `${equipmentPred.probability}%`;

      const equipmentBadge = document.getElementById('equipment-badge');
      equipmentBadge.textContent = equipmentPred.risk_level;
      equipmentBadge.className = `risk-badge ${equipmentPred.risk_level.toLowerCase()}`;

      const equipmentProgress = document.getElementById('equipment-progress');
      equipmentProgress.style.width = `${equipmentPred.probability}%`;
      equipmentProgress.className = `progress-fill ${equipmentPred.risk_level.toLowerCase()}`;

      // Flight Cancellation Results
      const cancellationPred = result.predictions.flight_cancellation;
      document.getElementById('cancellation-prediction').textContent = cancellationPred.prediction;
      document.getElementById('cancellation-probability').textContent = `${cancellationPred.probability}%`;

      const cancellationBadge = document.getElementById('cancellation-badge');
      cancellationBadge.textContent = cancellationPred.risk_level;
      cancellationBadge.className = `risk-badge ${cancellationPred.risk_level.toLowerCase()}`;

      const cancellationProgress = document.getElementById('cancellation-progress');
      cancellationProgress.style.width = `${cancellationPred.probability}%`;
      cancellationProgress.className = `progress-fill ${cancellationPred.risk_level.toLowerCase()}`;

      // Flight Summary
      if (result.input_summary) {
          document.getElementById('summary-aircraft').textContent = result.input_summary.aircraft;
          document.getElementById('summary-route').textContent = result.input_summary.route;
          document.getElementById('summary-maintenance').textContent = result.input_summary.maintenance;
      }

      // Scroll to results
      resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /**
   * Utility Functions
   */

  // Format numbers with commas
  function formatNumber(num) {
      return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  // Calculate risk color based on percentage
  function getRiskColor(percentage) {
      if (percentage < 30) return 'var(--success-green)';
      if (percentage < 60) return 'var(--warning-yellow)';
      return 'var(--danger-red)';
  }

  // Log application info
  console.log('%c Aircraft Prediction System ', 'background: #0078D4; color: white; font-size: 16px; padding: 10px;');
  console.log('%c Professional Data Analytics Platform ', 'color: #0078D4; font-size: 12px;');
  console.log('%c Powered by XGBoost Machine Learning ', 'color: #6B7280; font-size: 10px;');
  /**
   * ================================================================================
   * ANALYTICS FUNCTIONALITY
   * ================================================================================
   */

  /**
   * Load and Display Analytics Data
   */
   async function loadAnalytics() {
      console.log('loadAnalytics() called');
      const loadingElement = document.getElementById('analytics-loading');
      const contentElement = document.getElementById('analytics-content');

      console.log('Loading element:', loadingElement);
      console.log('Content element:', contentElement);

      try {
          console.log('Fetching analytics from API...');
          const response = await fetch(`${API_BASE_URL}/api/analytics`);
          console.log('Response received:', response);

          const result = await response.json();
          console.log('Result parsed:', result);

          if (result.success) {
              console.log('Success! Processing data...');
              const data = result.data;

              // Hide loading, show content
              loadingElement.style.display = 'none';
              contentElement.style.display = 'block';
              console.log('Loading hidden, content shown');

              // Populate Model Performance Metrics
              console.log('Populating model metrics...');
              populateModelMetrics(data.model_performance);

              // Populate Historical Data
              console.log('Populating historical data...');
              populateHistoricalData(data.historical_data);

              // Create all charts
              console.log('Creating charts...');
              createAnalyticsCharts(data);

              console.log('Analytics fully loaded!');
          } else {
              console.error('API returned success=false:', result);
              loadingElement.innerHTML = `
                  <div class="card-body" style="text-align: center; padding: 3rem;">
                      <p style="color: var(--danger-red);">Failed to load analytics: ${result.error}</p>
                  </div>
              `;
          }
      } catch (error) {
          console.error('Analytics loading error:', error);
          loadingElement.innerHTML = `
              <div class="card-body" style="text-align: center; padding: 3rem;">
                  <p style="color: var(--danger-red);">Failed to load analytics. Please ensure the API is running.</p>
              </div>
          `;
      }
  }

  /**
   * Populate Model Performance Metrics
   */
  function populateModelMetrics(modelPerformance) {
      const equip = modelPerformance.equipment_failure;
      const cancel = modelPerformance.flight_cancellation;

      // Equipment Failure Metrics
      document.getElementById('equip-roc-auc').textContent = (equip.roc_auc * 100).toFixed(2) + '%';
      document.getElementById('equip-accuracy').textContent = (equip.accuracy * 100).toFixed(2) + '%';
      document.getElementById('equip-precision').textContent = (equip.precision * 100).toFixed(2) + '%';
      document.getElementById('equip-recall').textContent = (equip.recall * 100).toFixed(2) + '%';
      document.getElementById('equip-f1').textContent = (equip.f1_score * 100).toFixed(2) + '%';

      // Flight Cancellation Metrics
      document.getElementById('cancel-roc-auc').textContent = (cancel.roc_auc * 100).toFixed(2) + '%';
      document.getElementById('cancel-accuracy').textContent = (cancel.accuracy * 100).toFixed(2) + '%';
      document.getElementById('cancel-precision').textContent = (cancel.precision * 100).toFixed(2) + '%';
      document.getElementById('cancel-recall').textContent = (cancel.recall * 100).toFixed(2) + '%';
      document.getElementById('cancel-f1').textContent = (cancel.f1_score * 100).toFixed(2) + '%';
  }

  /**
   * Populate Historical Data Statistics
   */
  function populateHistoricalData(historicalData) {
      document.getElementById('total-flights').textContent = historicalData.total_flights.toLocaleString();
      document.getElementById('total-failures').textContent = historicalData.equipment_failures.toLocaleString();
      document.getElementById('total-cancellations').textContent = historicalData.flight_cancellations.toLocaleString();

      const successRate = 100 - historicalData.equipment_failure_rate;
      document.getElementById('success-rate').textContent = successRate.toFixed(1) + '%';

      // Populate risky routes table
      const tbody = document.querySelector('#risky-routes-table tbody');
      tbody.innerHTML = '';

      historicalData.top_risky_routes.forEach(route => {
          const row = document.createElement('tr');
          row.innerHTML = `
              <td><strong>${route.route}</strong></td>
              <td>${route.total_flights.toLocaleString()}</td>
              <td>${route.failures}</td>
              <td><span style="color: ${route.failure_rate > 25 ? 'var(--danger-red)' : route.failure_rate > 15 ? 'var(--warning-yellow)' : 'var(--success-green)'}; font-weight: 600;">${route.failure_rate.toFixed(2)}%</span></td>
          `;
          tbody.appendChild(row);
      });
  }

  /**
   * Create All Analytics Charts
   */
  function createAnalyticsCharts(data) {
      // Confusion Matrix - Equipment Failure
      createConfusionMatrix('equip-confusion-chart', data.model_performance.equipment_failure.confusion_matrix, 'Equipment Failure');

      // Confusion Matrix - Flight Cancellation
      createConfusionMatrix('cancel-confusion-chart', data.model_performance.flight_cancellation.confusion_matrix, 'Flight Cancellation');

      // Risk Distribution Chart
      createRiskDistributionChart(data.historical_data.risk_distribution);

      // Aircraft Type Chart
      createAircraftTypeChart(data.historical_data.aircraft_types);

      // Weather Chart
      createWeatherChart(data.historical_data.weather_conditions);

      // Maintenance Chart
      createMaintenanceChart(data.historical_data.maintenance_impact);
  }

  /**
   * Create Confusion Matrix Chart
   */
  function createConfusionMatrix(canvasId, cm, title) {
      const ctx = document.getElementById(canvasId).getContext('2d');

      new Chart(ctx, {
          type: 'bar',
          data: {
              labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
              datasets: [{
                  label: 'Count',
                  data: [cm.true_negative, cm.false_positive, cm.false_negative, cm.true_positive],
                  backgroundColor: [
                      'rgba(16, 185, 129, 0.8)',  // Green for TN
                      'rgba(245, 158, 11, 0.8)',  // Yellow for FP
                      'rgba(239, 68, 68, 0.8)',   // Red for FN
                      'rgba(59, 130, 246, 0.8)'   // Blue for TP
                  ],
                  borderColor: [
                      'rgb(16, 185, 129)',
                      'rgb(245, 158, 11)',
                      'rgb(239, 68, 68)',
                      'rgb(59, 130, 246)'
                  ],
                  borderWidth: 2
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              aspectRatio: 2.5,
              plugins: {
                  legend: { display: false },
                  title: { display: false }
              },
              scales: {
                  y: {
                      beginAtZero: true,
                      ticks: { font: { size: 10 } }
                  },
                  x: {
                      ticks: { font: { size: 9 } }
                  }
              }
          }
      });
  }

  /**
   * Create Risk Distribution Pie Chart
   */
  function createRiskDistributionChart(riskDist) {
      const ctx = document.getElementById('risk-distribution-chart').getContext('2d');

      new Chart(ctx, {
          type: 'doughnut',
          data: {
              labels: ['Low Risk', 'Medium Risk', 'High Risk'],
              datasets: [{
                  data: [riskDist.low, riskDist.medium, riskDist.high],
                  backgroundColor: [
                      'rgba(16, 185, 129, 0.8)',
                      'rgba(245, 158, 11, 0.8)',
                      'rgba(239, 68, 68, 0.8)'
                  ],
                  borderColor: ['rgb(16, 185, 129)', 'rgb(245, 158, 11)', 'rgb(239, 68, 68)'],
                  borderWidth: 2
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              aspectRatio: 1.5,
              plugins: {
                  legend: {
                      position: 'bottom',
                      labels: { font: { size: 10 } }
                  }
              }
          }
      });
  }

  /**
   * Create Aircraft Type Bar Chart
   */
  function createAircraftTypeChart(aircraftTypes) {
      const ctx = document.getElementById('aircraft-type-chart').getContext('2d');

      const sortedTypes = aircraftTypes.sort((a, b) => b.failure_rate - a.failure_rate).slice(0, 5);

      new Chart(ctx, {
          type: 'bar',
          data: {
              labels: sortedTypes.map(a => a.type),
              datasets: [{
                  label: 'Failure Rate (%)',
                  data: sortedTypes.map(a => a.failure_rate),
                  backgroundColor: 'rgba(0, 120, 212, 0.8)',
                  borderColor: 'rgb(0, 120, 212)',
                  borderWidth: 2
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              aspectRatio: 2,
              plugins: {
                  legend: { display: false }
              },
              scales: {
                  y: {
                      beginAtZero: true,
                      ticks: {
                          callback: value => value + '%',
                          font: { size: 10 }
                      }
                  },
                  x: {
                      ticks: { font: { size: 9 } }
                  }
              }
          }
      });
  }

  /**
   * Create Weather Condition Chart
   */
  function createWeatherChart(weatherConditions) {
      const ctx = document.getElementById('weather-chart').getContext('2d');

      new Chart(ctx, {
          type: 'bar',
          data: {
              labels: weatherConditions.map(w => w.condition),
              datasets: [{
                  label: 'Failure Rate (%)',
                  data: weatherConditions.map(w => w.failure_rate),
                  backgroundColor: 'rgba(0, 188, 242, 0.8)',
                  borderColor: 'rgb(0, 188, 242)',
                  borderWidth: 2
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              aspectRatio: 2,
              indexAxis: 'y',
              plugins: {
                  legend: { display: false }
              },
              scales: {
                  x: {
                      beginAtZero: true,
                      ticks: {
                          callback: value => value + '%',
                          font: { size: 10 }
                      }
                  },
                  y: {
                      ticks: { font: { size: 9 } }
                  }
              }
          }
      });
  }

  /**
   * Create Maintenance Impact Line Chart
   */
  function createMaintenanceChart(maintenanceImpact) {
      const ctx = document.getElementById('maintenance-chart').getContext('2d');

      new Chart(ctx, {
          type: 'line',
          data: {
              labels: maintenanceImpact.map(m => m.days_range + ' days'),
              datasets: [{
                  label: 'Failure Rate (%)',
                  data: maintenanceImpact.map(m => m.failure_rate),
                  backgroundColor: 'rgba(239, 68, 68, 0.2)',
                  borderColor: 'rgb(239, 68, 68)',
                  borderWidth: 3,
                  fill: true,
                  tension: 0.4
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              aspectRatio: 2,
              plugins: {
                  legend: { display: false }
              },
              scales: {
                  y: {
                      beginAtZero: true,
                      ticks: {
                          callback: value => value + '%',
                          font: { size: 10 }
                      }
                  },
                  x: {
                      ticks: { font: { size: 9 } }
                  }
              }
          }
      });
  }
   
  /**
   * ================================================================================
   * BATCH PROCESSING FUNCTIONALITY
   * ================================================================================
   */

  // Global variables to store batch data
  let uploadedFileData = null;
  let batchResults = null;
  let batchProcessingInitialized = false; // Flag to prevent duplicate initialization

  /**
   * Setup all event listeners for batch processing
   */
  function setupBatchProcessing() {
      // Prevent duplicate initialization
      if (batchProcessingInitialized) {
          console.log('Batch processing already initialized, skipping...');
          return;
      }

      // Get all required elements
      const uploadArea = document.getElementById('upload-area');
      const fileInput = document.getElementById('csv-file-input');
      const browseBtn = document.getElementById('browse-btn');
      const removeFileBtn = document.getElementById('remove-file-btn');
      const processBatchBtn = document.getElementById('process-batch-btn');
      const downloadTemplateBtn = document.getElementById('download-template');
      const exportResultsBtn = document.getElementById('export-results-btn');

      // Safety check - if ANY element is missing, exit silently
      if (!uploadArea || !fileInput || !browseBtn || !removeFileBtn ||
          !processBatchBtn || !downloadTemplateBtn || !exportResultsBtn) {
          console.log('Batch processing elements not ready yet');
          return;
      }

      console.log('Setting up batch processing...');
      batchProcessingInitialized = true; // Mark as initialized

      // Browse button - opens file picker
      browseBtn.addEventListener('click', () => {
          console.log('Browse button clicked');
          fileInput.click();
      });

      // File input change event
      fileInput.addEventListener('change', (e) => {
          console.log('File input changed, files:', e.target.files.length);
          if (e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
          }
      });

      // Drag and drop events
      uploadArea.addEventListener('dragover', (e) => {
          e.preventDefault();
          uploadArea.style.borderColor = 'var(--primary-blue)';
          uploadArea.style.background = 'var(--gray-50)';
      });

      uploadArea.addEventListener('dragleave', (e) => {
          e.preventDefault();
          uploadArea.style.borderColor = 'var(--gray-300)';
          uploadArea.style.background = 'var(--white)';
      });

      uploadArea.addEventListener('drop', (e) => {
          e.preventDefault();
          uploadArea.style.borderColor = 'var(--gray-300)';
          uploadArea.style.background = 'var(--white)';

          const files = e.dataTransfer.files;
          if (files.length > 0 && files[0].name.endsWith('.csv')) {
              handleFileUpload(files[0]);
          }
      });

      // Remove file button
      removeFileBtn.addEventListener('click', resetBatchUpload);

      // Process batch button
      processBatchBtn.addEventListener('click', processBatchPredictions);

      // Download template button
      downloadTemplateBtn.addEventListener('click', downloadCSVTemplate);

      // Export results button
      exportResultsBtn.addEventListener('click', exportResultsToCSV);

      console.log('Batch processing setup complete!');
  }
   /**
   * Handle CSV file upload
   */
  function handleFileUpload(file) {
      console.log('Handling file upload:', file.name);

      if (!file.name.endsWith('.csv')) {
          alert('Please upload a CSV file');
          return;
      }

      const reader = new FileReader();

      reader.onload = function(e) {
          try {
              const csvText = e.target.result;
              const parsedData = parseCSV(csvText);

              if (parsedData && parsedData.length > 0) {
                  uploadedFileData = parsedData;
                  console.log('Parsed data:', parsedData.length, 'rows');

                  // Verify elements exist before updating
                  const uploadArea = document.getElementById('upload-area');
                  const fileSelectedInfo = document.getElementById('file-selected-info');
                  const fileName = document.getElementById('file-name');
                  const fileStats = document.getElementById('file-stats');

                  if (!uploadArea || !fileSelectedInfo || !fileName || !fileStats) {
                      console.error('Required UI elements not found');
                      alert('UI elements not ready. Please refresh the page.');
                      return;
                  }

                  // Update UI
                  uploadArea.style.display = 'none';
                  fileSelectedInfo.style.display = 'block';
                  fileName.textContent = file.name;
                  fileStats.textContent = `${parsedData.length} flights loaded`;

                  console.log('UI updated successfully');
              } else {
                  alert('Failed to parse CSV file. Please check the format.');
              }
          } catch (error) {
              console.error('Error in file upload:', error);
              alert('Error processing file: ' + error.message);
          }
      };

      reader.onerror = function() {
          alert('Error reading file');
      };

      reader.readAsText(file);
  }

  /**
   * Parse CSV text to JSON array
   */
  function parseCSV(csvText) {
      const lines = csvText.trim().split('\n');
      if (lines.length < 2) return null;

      const headers = lines[0].split(',').map(h => h.trim());
      const data = [];

      for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          const row = {};

          headers.forEach((header, index) => {
              const value = values[index];
              // Convert numeric values
              if (!isNaN(value) && value !== '') {
                  row[header] = parseFloat(value);
              } else {
                  row[header] = value;
              }
          });

          data.push(row);
      }

      return data;
  }

  /**
   * Process batch predictions
   */
  async function processBatchPredictions() {
      if (!uploadedFileData || uploadedFileData.length === 0) {
          alert('No data to process');
          return;
      }

      // Show processing status
      document.getElementById('file-selected-info').style.display = 'none';
      document.getElementById('batch-processing-status').style.display = 'block';
      document.getElementById('processing-progress').textContent =
          `Processing ${uploadedFileData.length} flights...`;

      // Trigger airplane takeoff animation
      triggerAirplaneTakeoff();

      try {
          const response = await fetch(`${API_BASE_URL}/api/batch-predict`, {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify(uploadedFileData)
          });

          if (!response.ok) {
              throw new Error('Batch prediction failed');
          }

          const result = await response.json();

          if (result.success) {
              batchResults = result;
              displayBatchResults(result);
          } else {
              throw new Error(result.error || 'Prediction failed');
          }

      } catch (error) {
          console.error('Batch processing error:', error);
          alert('Error processing batch: ' + error.message);
          resetBatchUpload();
      }
  }

  /**
   * Display batch prediction results
   */
  function displayBatchResults(result) {
      // Hide processing status
      document.getElementById('batch-processing-status').style.display = 'none';

      // Show results container
      document.getElementById('batch-results-container').style.display = 'block';

      // Update summary stats
      document.getElementById('batch-total-flights').textContent = result.total_flights;
      document.getElementById('batch-total-failures').textContent = result.summary.total_failures_predicted;
      document.getElementById('batch-total-cancellations').textContent = result.summary.total_cancellations_predicted;

      const successRate = ((result.total_flights - result.summary.total_failures_predicted - result.summary.total_cancellations_predicted) / result.total_flights * 100).toFixed(1);
      document.getElementById('batch-success-rate').textContent = successRate + '%';

      // Populate results table
      const tbody = document.getElementById('batch-results-table').querySelector('tbody');
      tbody.innerHTML = '';

      result.predictions.forEach((pred, index) => {
          const originalData = uploadedFileData[index];
          const row = document.createElement('tr');

          // Determine risk badge classes
          const equipRisk = pred.equipment_failure.prediction === 'Failure' ? 'danger' : 'success';
          const cancelRisk = pred.flight_cancellation.prediction === 'Cancelled' ? 'warning' : 'success';

          row.innerHTML = `
              <td>${index + 1}</td>
              <td>${originalData.aircraft_type || 'N/A'}</td>
              <td>${originalData.origin || 'N/A'} → ${originalData.destination || 'N/A'}</td>
              <td><span class="risk-badge ${equipRisk}">${pred.equipment_failure.prediction}</span></td>
              <td>${pred.equipment_failure.probability}%</td>
              <td><span class="risk-badge ${cancelRisk}">${pred.flight_cancellation.prediction}</span></td>
              <td>${pred.flight_cancellation.probability}%</td>
          `;

          tbody.appendChild(row);
      });
  }

  /**
   * Download CSV template
   */
  function downloadCSVTemplate() {
      const headers = [
          'aircraft_type', 'aircraft_age', 'operational_hours',
          'days_since_maintenance', 'maintenance_count', 'airline',
          'origin', 'destination', 'distance', 'flight_duration',
          'weather_condition', 'weather_severity', 'engine_temperature',
          'oil_pressure', 'vibration_level', 'fuel_consumption',
          'hydraulic_pressure', 'cabin_pressure', 'previous_delays',
          'crew_experience'
      ];

      const exampleRow = [
          'Boeing 737', '15', '25000', '120', '25', 'AirGlobal',
          'JFK', 'LAX', '2475', '5.2', 'Clear', '1', '295.5',
          '42.3', '12.8', '3500', '3050', '11.2', '2', '12.5'
      ];

      const csvContent = headers.join(',') + '\n' + exampleRow.join(',');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'flight_data_template.csv';
      a.click();
      window.URL.revokeObjectURL(url);
  }

  /**
   * Export results to CSV
   */
  function exportResultsToCSV() {
      if (!batchResults || !uploadedFileData) {
          alert('No results to export');
          return;
      }

      // Create CSV headers
      const headers = [
          'Flight_Index',
          'Aircraft_Type',
          'Route',
          'Equipment_Failure_Prediction',
          'Equipment_Failure_Probability',
          'Flight_Cancellation_Prediction',
          'Flight_Cancellation_Probability'
      ];

      // Create CSV rows
      const rows = batchResults.predictions.map((pred, index) => {
          const original = uploadedFileData[index];
          return [
              index + 1,
              original.aircraft_type || 'N/A',
              `${original.origin || 'N/A'} → ${original.destination || 'N/A'}`,
              pred.equipment_failure.prediction,
              pred.equipment_failure.probability + '%',
              pred.flight_cancellation.prediction,
              pred.flight_cancellation.probability + '%'
          ].join(',');
      });

      const csvContent = headers.join(',') + '\n' + rows.join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'batch_prediction_results.csv';
      a.click();
      window.URL.revokeObjectURL(url);
  }

  /**
   * Reset batch upload UI
   */
  function resetBatchUpload() {
      uploadedFileData = null;
      batchResults = null;

      // Safely reset UI elements with null checks
      const uploadArea = document.getElementById('upload-area');
      const fileSelectedInfo = document.getElementById('file-selected-info');
      const batchProcessingStatus = document.getElementById('batch-processing-status');
      const batchResultsContainer = document.getElementById('batch-results-container');
      const csvFileInput = document.getElementById('csv-file-input');

      if (uploadArea) uploadArea.style.display = 'flex';
      if (fileSelectedInfo) fileSelectedInfo.style.display = 'none';
      if (batchProcessingStatus) batchProcessingStatus.style.display = 'none';
      if (batchResultsContainer) batchResultsContainer.style.display = 'none';
      if (csvFileInput) csvFileInput.value = '';

      console.log('Batch upload reset complete');
  }

 

/**
 * ================================================================================
 * AIRPLANE TAKEOFF ANIMATION
 * ================================================================================
 */

/**
 * Trigger airplane takeoff animation
 */
function triggerAirplaneTakeoff() {
    const airplane = document.getElementById('airplane-animation');

    if (!airplane) {
        console.warn('Airplane animation element not found');
        return;
    }

    // Remove existing animation class if present
    airplane.classList.remove('taking-off');

    // Force reflow to restart animation
    void airplane.offsetWidth;

    // Add animation class
    airplane.classList.add('taking-off');

    // Remove class after animation completes (3 seconds)
    setTimeout(() => {
        airplane.classList.remove('taking-off');
    }, 3000);
}

// VERSION CHECK - BATCH PROCESSING FIX v3.0 - DUPLICATE LISTENER FIX + AIRPLANE ANIMATION
console.log('%c BATCH PROCESSING FIX v3.0 + AIRPLANE ANIMATION LOADED ', 'background: purple; color: white; font-size: 14px; padding: 5px;');
window.APP_VERSION = '3.0-BATCH-FIX-AIRPLANE';
