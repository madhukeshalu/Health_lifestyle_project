// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    
    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const requiredFields = form.querySelectorAll('[required]');
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('is-invalid');
                    
                    // Add error message
                    if (!field.nextElementSibling || !field.nextElementSibling.classList.contains('invalid-feedback')) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'invalid-feedback';
                        errorDiv.textContent = 'This field is required';
                        field.parentNode.appendChild(errorDiv);
                    }
                } else {
                    field.classList.remove('is-invalid');
                    field.classList.add('is-valid');
                    
                    // Remove error message
                    const errorDiv = field.nextElementSibling;
                    if (errorDiv && errorDiv.classList.contains('invalid-feedback')) {
                        errorDiv.remove();
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                
                // Show alert
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-danger alert-dismissible fade show';
                alertDiv.innerHTML = `
                    <strong>Missing Information!</strong> Please fill in all required fields.
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;
                form.prepend(alertDiv);
            }
        });
    });
    
    // Real-time BMI calculator (if on prediction page)
    const heightInput = document.querySelector('input[name="height_cm"]');
    const weightInput = document.querySelector('input[name="weight_kg"]');
    
    if (heightInput && weightInput) {
        function calculateBMI() {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                
                // Show BMI preview
                let bmiPreview = document.getElementById('bmiPreview');
                if (!bmiPreview) {
                    bmiPreview = document.createElement('div');
                    bmiPreview.id = 'bmiPreview';
                    bmiPreview.className = 'alert alert-info mt-2';
                    weightInput.parentNode.appendChild(bmiPreview);
                }
                
                let category = '';
                let color = 'info';
                
                if (bmi < 18.5) {
                    category = 'Underweight';
                    color = 'info';
                } else if (bmi < 25) {
                    category = 'Normal';
                    color = 'success';
                } else if (bmi < 30) {
                    category = 'Overweight';
                    color = 'warning';
                } else {
                    category = 'Obese';
                    color = 'danger';
                }
                
                bmiPreview.innerHTML = `
                    Estimated BMI: <strong>${bmi.toFixed(1)}</strong> 
                    <span class="badge bg-${color}">${category}</span>
                `;
            }
        }
        
        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }
    
    // Copy session ID
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sessionId = this.getAttribute('data-session');
            navigator.clipboard.writeText(sessionId).then(() => {
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    this.innerHTML = originalText;
                }, 2000);
            });
        });
    });
    
    // Print report
    const printBtn = document.getElementById('printReport');
    if (printBtn) {
        printBtn.addEventListener('click', function() {
            window.print();
        });
    }
});