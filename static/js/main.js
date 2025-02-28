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
    
    // Sample data functionality - using the API route
    const loadSampleDataButtons = document.querySelectorAll('#loadSampleData');
    if (loadSampleDataButtons.length > 0) {
        loadSampleDataButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Show loading status
                const originalText = this.innerHTML;
                this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
                this.disabled = true;
                
                // Call API to load sample data
                fetch('/api/load_sample_data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        createToast('Error: ' + data.error, 'danger');
                        this.innerHTML = originalText;
                        this.disabled = false;
                    } else {
                        createToast('Sample data loaded successfully!', 'success');
                        // Redirect to EDA page
                        window.location.href = '/eda';
                    }
                })
                .catch(error => {
                    createToast('Error: ' + error.message, 'danger');
                    this.innerHTML = originalText;
                    this.disabled = false;
                });
            });
        });
    }
    
    // Function to create a toast notification
    window.createToast = function(message, type = 'info') {
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
    };
    
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