    // File Upload Handling
    function handleFileUpload() {
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fastaInput');
        const spinner = document.getElementById('loading-spinner');
        const resultsContainer = document.getElementById('resultsContainer');
    
        if (fileInput.files.length > 0) {
            spinner.style.display = 'block';
            resultsContainer.innerHTML = '';
    
            const formData = new FormData(form);
            
            fetch(window.location.href, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                spinner.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                    updateDescriptorOptions(data.sequence_type);  // Directly update descriptors
                }
            })
            .catch(error => {
                spinner.style.display = 'none';
                showError('An error occurred during analysis');
            });
        }
    }
    
    // Results Display
    function showResults(data) {
        const template = document.getElementById('resultTemplate').content.cloneNode(true);
        
        template.querySelector('#typeBadge').textContent = data.sequence_type;
        template.querySelector('#sequenceTitle').textContent = data.sequence_id || 'Unnamed Sequence';
        template.querySelector('#sequencePreview').textContent = data.sequence_preview;
    
        const sequenceTypeInput = document.getElementById('sequenceType');
        sequenceTypeInput.value = data.sequence_type;
        document.getElementById('resultsContainer').appendChild(template);
    }
    
    // Descriptor Options Management
    function updateDescriptorOptions(sequenceType) {
        const descriptorSelect = document.getElementById('descriptorSelect');
        descriptorSelect.innerHTML = '<option value="" disabled selected>Select descriptor</option>';
    
        const options = {
            'Protein': [
                { value: 'AAC', text: 'Amino Acid Composition (AAC)' },
                { value: 'PAAC', text: 'Pseudo Amino Acid Composition (PAAC)' },
                { value: 'EAAC', text: 'Enhanced Amino Acids Content (EAAC)' },
                { value: 'CKSAAP', text: 'Composition of k-spaced Amino Acid Pairs (CKSAAP)' },
                { value: 'DPC', text: 'Di-Peptide Composition (DPC)' },
                { value: 'DDE', text: 'Dipeptide Deviation from Expected Mean (DDE)' },
                { value: 'TPC', text: 'Tripeptide Composition (TPC)' },
                { value: 'GAAC', text: 'Ground Amino Acid Composition (GAAC)' },
                { value: 'GAAC_Grouped', text: 'Grouped Amino Acid Composition (GAAC)' }
            ],
    
            'DNA': [
                { value: 'Kmer', 'text': 'The occurrence frequencies of k neighboring nucleic acids(Kmer)' },
                { value: 'RCKmer', 'text': 'Reverse Compliment Kmer(RCKmer)' },
                { value: 'Mismatch', 'text': 'Mismatch profile(Mismatch)' },
                { value: 'Subsequence', 'text': 'Subsequence profile' },
                { value: 'NAC', 'text': 'Nucleic Acid Composition(NAC)' },
                { value: 'ANF', 'text': 'Accumulated Nucleotide Frequency(ANF)' },
                { value: 'ENAC', 'text': 'Enhanced Nucleic Acid Composition(ENAC)' }
            ],
     
            'RNA': [
                { value: 'Kmer', 'text': 'The occurrence frequencies of k neighboring nucleic acids(Kmer)' },
                { value: 'Mismatch', 'text': 'Mismatch profile' },
                { value: 'Subsequence', 'text': 'Subsequence profile' },
                { value: 'NAC', 'text': 'Nucleic Acid Composition(NAC)' },
                { value: 'ENAC', 'text': 'Enhanced Nucleic Acid Composition(ENAC)' },
                { value: 'ANF', 'text': 'Accumulated Nucleotide Frequency(ANF)' },
                { value: 'NCP', 'text': 'Nucleotide Chemical Property(NCP)' },
                { value: 'PSTNPss', 'text': 'Position-specific trinucleotide propensity based on single strand(PSTNPss)' }
    ]
        };
        
    
        const group = document.createElement('optgroup');
        if (options[sequenceType]) {
            group.label = `${sequenceType} Analysis`;
            options[sequenceType].forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                group.appendChild(option);
            });
        } else {
            group.label = 'Error';
            const option = document.createElement('option');
            option.textContent = 'Unknown sequence type - Please verify your input';
            option.disabled = true;
            group.appendChild(option);
        }
        
        descriptorSelect.appendChild(group);
    }
    
    // Update parameters on select Descriptor
    document.addEventListener("DOMContentLoaded", function () {
        const descriptorSelect = document.getElementById("descriptorSelect");
        const parameterContainer = document.getElementById("parameterContainer");
    
        // Descriptor parameters mapping
        const descriptorParameters = {
            "CTD": [
                { label: "K-space Number", type: "number", id: "k_space", default: 3 }
            ],
            "AAC": [
                { label: "Normalization", type: "checkbox", id: "normalize", default: true }
            ],
            "PAAC": [
                { label: "Lambda", type: "number", id: "lambda", default: 4 },
                { label: "Weight Factor", type: "number", id: "weight", default: 0.05 }
            ],
            "Kmer": [
                { label: "K-mer Length", type: "number", id: "kmer_length", default: 3 }
            ],
            "DPC": [
                { label: "Normalization", type: "checkbox", id: "normalize_dpc", default: true }
            ],
            "EAAC": [
                { label: "Sliding Window Size", type: "number", id: "window_size", default: 5 }
            ],
            "CKSAAP": [
                { label: "K-space number", type: "number", id: "K-space number", default: 3 }
            ],
    
        };
    
        // Function to update parameters
        function updateParameters() {
            const selectedDescriptor = descriptorSelect.value;
            parameterContainer.innerHTML = ""; // Clear previous parameters
    
            if (descriptorParameters[selectedDescriptor]) {
                descriptorParameters[selectedDescriptor].forEach(param => {
                    const inputGroup = document.createElement("div");
                    inputGroup.classList.add("mb-2");
    
                    const label = document.createElement("label");
                    label.classList.add("form-label");
                    label.textContent = param.label;
    
                    let input;
                    if (param.type === "checkbox") {
                        input = document.createElement("input");
                        input.type = "checkbox";
                        input.classList.add("form-check-input");
                        input.checked = param.default;
                    } else {
                        input = document.createElement("input");
                        input.type = param.type;
                        input.classList.add("form-control");
                        input.value = param.default;
                    }
    
                    input.id = param.id;
                    input.name = param.id;
    
                    inputGroup.appendChild(label);
                    inputGroup.appendChild(input);
                    parameterContainer.appendChild(inputGroup);
                });
            } else {
                parameterContainer.innerHTML = '<div class="alert alert-info">No parameters required for this descriptor.</div>';
            }
        }
    
        // Event listener for descriptor selection change
        descriptorSelect.addEventListener("change", updateParameters);
    });
    
    // start Analysis
    document.addEventListener("DOMContentLoaded", function () {
        const startAnalysisBtn = document.getElementById("startAnalysis");
        const loadingAnimation = document.getElementById("loadingAnimation");
        let extractedData = []; // Store extracted features
        let currentPage = 0;
        const rowsPerPage = 10;
    
        startAnalysisBtn.addEventListener("click", function () {
            const sequenceType = document.getElementById("sequenceType").value;
            const descriptor = document.getElementById("descriptorSelect").value;
            const fileInput = document.getElementById("fastaInput");
    
            if (!fileInput.files.length) {
                showError("Please upload a FASTA file first");
                return;
            }
    
            if (!descriptor) {
                showError("Please select a descriptor");
                return;
            }
    
            loadingAnimation.style.display = "block";
    
            const formData = new FormData();
            formData.append("fasta_file", fileInput.files[0]);
            formData.append("descriptor", descriptor);
            formData.append("sequence_type", sequenceType);
    
            // Add Parameters
            const parameterInputs = document.querySelectorAll("#parameterContainer input");
            parameterInputs.forEach(input => {
                
                if (input.type === "checkbox") {
                    formData.append('checked_para', input.checked);
                } else {
                    formData.append('parameter', input.value);
                }
            });
    
            fetch("/feature_extraction/analyze/", {
                method: "POST",
                body: formData,
                headers: {
                    "X-CSRFToken": getCookie("csrftoken"),
                    "X-Requested-With": "XMLHttpRequest"
                }
            })
                .then(response => response.json())
                .then(data => {
                    loadingAnimation.style.display = "none";
                    if (data.error) {
                        showError(data.error);
                    } else {
                        extractedData = data.csv_data.split("\n").map(row => row.split(",")); // Convert CSV to array
                        currentPage = 0;
                        updateTable();
                        document.getElementById("result_analysis_section").style.display = "block"; // Show Analysis Section
                    }
                })
                .catch(error => {
                    loadingAnimation.style.display = "none";
                    showError("An error occurred during analysis");
                    console.error(error);
                });
        });
    
        function updateTable() {
            const iframe = document.getElementById("featureTable");
            const doc = iframe.contentDocument || iframe.contentWindow.document;
            doc.open();
            doc.write("<html><head><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'></head><body>");
    
            if (extractedData.length === 0) {
                doc.write("<p>No extracted features available.</p>");
            } else {
                let tableHTML = "<table class='table table-striped table-bordered'><thead><tr>";
                extractedData[0].forEach(header => tableHTML += `<th>${header}</th>`);
                tableHTML += "</tr></thead><tbody>";
    
                let startRow = currentPage * rowsPerPage + 1;
                let endRow = Math.min(startRow + rowsPerPage, extractedData.length);
    
                for (let i = startRow; i < endRow; i++) {
                    tableHTML += "<tr>";
                    extractedData[i].forEach(cell => tableHTML += `<td>${cell}</td>`);
                    tableHTML += "</tr>";
                }
                tableHTML += "</tbody></table>";
                doc.write(tableHTML);
            }
    
            doc.write("</body></html>");
            doc.close();
    
            document.getElementById("prevBtn").disabled = currentPage === 0;
            document.getElementById("nextBtn").disabled = (currentPage + 1) * rowsPerPage >= extractedData.length;
        }
    
        document.getElementById("prevBtn").addEventListener("click", function () {
            if (currentPage > 0) {
                currentPage--;
                updateTable();
            }
        });
    
        document.getElementById("nextBtn").addEventListener("click", function () {
            if ((currentPage + 1) * rowsPerPage < extractedData.length) {
                currentPage++;
                updateTable();
            }
        });
    
        document.getElementById("downloadBtn").addEventListener("click", function () {
            if (extractedData.length === 0) return;
    
            let csvContent = extractedData.map(e => e.join(",")).join("\n");
            const blob = new Blob([csvContent], { type: "text/csv" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "extracted_features.csv";
            link.click();
        });
    
        function getCookie(name) {
            let cookieValue = null;
            if (document.cookie && document.cookie !== "") {
                const cookies = document.cookie.split(";");
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    if (cookie.startsWith(name + "=")) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
    
        function showError(message) {
            const errorDiv = document.createElement("div");
            errorDiv.className = "alert alert-danger";
            errorDiv.textContent = message;
            const container = document.getElementById("resultsContainer");
            container.prepend(errorDiv);
    
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }
    });
    
    // Modules button to send the CSV data before redirecting.
    document.getElementById('Module').addEventListener('click', function(e) {
        e.preventDefault();
        const url = this.href;
        const csvContent = extractedData.map(row => row.join(",")).join("\n");
    
        fetch('/feature_extraction/save_extracted_data/', {
            method: 'POST',
            body: JSON.stringify({ csv_data: csvContent }),
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
        })
        .then(response => {
            if (response.ok) {
                window.location.href = url;
            } else {
                alert('Error saving extracted features.');
            }
        })
        .catch(error => console.error('Error:', error));
    });
    
    