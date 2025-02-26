/**
 * Data Science Platform - Main JavaScript
 * Handles global interactions and utilities
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize all popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Sample data functionality
    const sampleDataLink = document.querySelector('.sample-data-link');
    if (sampleDataLink) {
        sampleDataLink.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Show modal for sample data selection
            const modal = new bootstrap.Modal(document.getElementById('sampleDataModal'));
            if (modal) {
                modal.show();
            } else {
                // Create a modal dynamically if it doesn't exist
                createSampleDataModal();
            }
        });
    }
    
    // Function to create sample data modal dynamically
    function createSampleDataModal() {
        const modalHTML = `
            <div class="modal fade" id="sampleDataModal" tabindex="-1" aria-labelledby="sampleDataModalLabel" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title" id="sampleDataModalLabel">Choose Sample Dataset</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div class="list-group">
                                <button type="button" class="list-group-item list-group-item-action" data-dataset="customer_churn">
                                    <strong>Customer Churn Dataset</strong>
                                    <p class="mb-0 text-muted">Sample dataset for churn prediction with customer demographics and usage metrics.</p>
                                </button>
                                <button type="button" class="list-group-item list-group-item-action" data-dataset="sales_forecasting">
                                    <strong>Retail Sales Dataset</strong>
                                    <p class="mb-0 text-muted">Time series data for sales forecasting with monthly sales figures.</p>
                                </button>
                                <button type="button" class="list-group-item list-group-item-action" data-dataset="marketing_campaign">
                                    <strong>Marketing Campaign Dataset</strong>
                                    <p class="mb-0 text-muted">Data for marketing campaign analysis with customer responses.</p>
                                </button>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Append the modal to the body
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Initialize the modal
        const modal = new bootstrap.Modal(document.getElementById('sampleDataModal'));
        modal.show();
        
        // Add event listeners to the dataset buttons
        document.querySelectorAll('[data-dataset]').forEach(button => {
            button.addEventListener('click', function() {
                const dataset = this.getAttribute('data-dataset');
                // Load the selected sample dataset
                loadSampleDataset(dataset);
                // Hide the modal
                modal.hide();
            });
        });
    }
    
    // Function to load a sample dataset
    function loadSampleDataset(dataset) {
        // Show loading indicator
        const loadingToast = createToast('Loading sample dataset...', 'info');
        
        // Simulated API request to load the sample dataset
        setTimeout(() => {
            // Redirect to the EDA page (in a real app, this would happen after actual data loading)
            window.location.href = '/eda';
        }, 1500);
    }
    
    // Function to create a toast notification
    function createToast(message, type = 'info') {
        // Check if toast container exists, create if not
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        // Create a new toast
        const toastId = 'toast-' + Date.now();
        const toastHTML = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header bg-${type} text-white">
                    <strong class="me-auto">Notification</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        // Add the toast to the container
        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        
        // Initialize and show the toast
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: 3000
        });
        toast.show();
        
        return toast;
    }
    
    // Global error handler for fetch requests
    window.handleFetchError = function(error, elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill"></i> Error: ${error.message}
                </div>
            `;
        }
        console.error('Fetch error:', error);
    };
    
    // Global utility to convert JSON to CSV
    window.jsonToCSV = function(jsonData, columns) {
        if (!jsonData || !jsonData.length) return '';
        
        // If columns are not provided, use the keys from the first object
        if (!columns) {
            columns = Object.keys(jsonData[0]);
        }
        
        // Create the header row
        let csv = columns.join(',') + '\n';
        
        // Add data rows
        jsonData.forEach(item => {
            const row = columns.map(column => {
                const value = item[column];
                // Handle commas, quotes, and undefined values
                if (value === null || value === undefined) {
                    return '';
                } else if (typeof value === 'string') {
                    // Escape quotes and wrap in quotes if contains comma or quote
                    const escaped = value.replace(/"/g, '""');
                    return escaped.includes(',') || escaped.includes('"') 
                        ? `"${escaped}"` 
                        : escaped;
                }
                return value;
            }).join(',');
            
            csv += row + '\n';
        });
        
        return csv;
    };
    
    // Function to download data as a file
    window.downloadFile = function(data, filename, mimeType) {
        const blob = new Blob([data], { type: mimeType });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    };
});
