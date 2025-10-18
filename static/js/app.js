(function () {
    let toastTimeout;

    function ensureToast() {
        let toast = document.querySelector('.clipboard-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'clipboard-toast';
            document.body.appendChild(toast);
        }
        return toast;
    }

    function showCopyToast(text) {
        const toast = ensureToast();
        toast.textContent = `Copied: ${text}`;
        toast.classList.add('show');
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => toast.classList.remove('show'), 1800);
    }

    function fallbackCopyText(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {
            document.execCommand('copy');
            showCopyToast(text);
        } finally {
            document.body.removeChild(textarea);
        }
    }

    function copyCellText(text) {
        if (!text) {
            return;
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => showCopyToast(text)).catch(() => {
                fallbackCopyText(text);
            });
        } else {
            fallbackCopyText(text);
        }
    }

    function attachCellCopy(tableId) {
        const table = document.getElementById(tableId);
        if (!table) {
            return;
        }

        if (table.dataset.copyBound === 'true') {
            return;
        }

        table.dataset.copyBound = 'true';

        table.addEventListener('click', (event) => {
            const cell = event.target.closest('td');
            if (!cell) {
                return;
            }
            const text = cell.innerText.trim();
            if (!text) {
                return;
            }
            copyCellText(text);
        });
    }

    window.renderNumeric = function renderNumeric(precision) {
        return function (data) {
            if (data === null || data === undefined || data === '') {
                return '';
            }
            const num = Number(data);
            if (Number.isNaN(num)) {
                return data;
            }
            return num.toFixed(precision);
        };
    };

    window.renderDataTable = function renderDataTable(tableId, columns, data, overrides = {}) {
        const tableSelector = `#${tableId}`;
        const columnDefs = columns.map((column) => {
            if (typeof column === 'string') {
                return { title: column, data: column };
            }
            return column;
        });

        const defaults = {
            data,
            columns: columnDefs,
            responsive: true,
            scrollX: true,
            scrollY: '60vh',
            scrollCollapse: true,
            paging: false,
            info: true,
            deferRender: true,
            autoWidth: false,
            lengthChange: false,
            dom: 'Bfrtip',
            buttons: ['copyHtml5', 'csvHtml5', 'excelHtml5'],
            language: {
                search: 'Search records:',
            },
        };

        const settings = Object.assign({}, defaults, overrides);

        if ($.fn.DataTable.isDataTable(tableSelector)) {
            const existing = $(tableSelector).DataTable();
            existing.clear();
            existing.rows.add(data);
            existing.columns.adjust().draw();
        } else {
            $(tableSelector).DataTable(settings);
        }

        attachCellCopy(tableId);
    };
})();
