/* =========================================================
   RESEARCHHUB FRONTEND
   ========================================================= */


document.addEventListener("DOMContentLoaded", function () {


    /* =====================================================
       UPLOAD FILE NAME
       ===================================================== */

    const fileInput =
        document.getElementById("fileInput");

    const fileText =
        document.getElementById("fileText");


    if (fileInput && fileText) {

        fileInput.addEventListener(
            "change",
            function () {

                if (this.files.length > 0) {

                    fileText.textContent =
                        this.files[0].name;

                } else {

                    fileText.textContent =
                        "Choose PDF file";

                }

            }
        );

    }


    /* =====================================================
       UPDATE FILE NAME
       ===================================================== */

    const updateFileInput =
        document.getElementById(
            "updateFileInput"
        );

    const updateFileText =
        document.getElementById(
            "updateFileText"
        );


    if (updateFileInput && updateFileText) {

        updateFileInput.addEventListener(
            "change",
            function () {

                if (this.files.length > 0) {

                    updateFileText.textContent =
                        this.files[0].name;

                } else {

                    updateFileText.textContent =
                        "Choose new PDF file";

                }

            }
        );

    }


    /* =====================================================
       UPLOAD BUTTON LOADING
       ===================================================== */

    const uploadForm =
        document.getElementById(
            "uploadForm"
        );


    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            function () {

                const button =
                    uploadForm.querySelector(
                        "button[type='submit']"
                    );


                if (button) {

                    button.disabled = true;

                    button.innerHTML =
                        "Processing...";

                }

            }
        );

    }


    /* =====================================================
       SEARCH FORM
       ===================================================== */

    const searchForm =
        document.getElementById(
            "searchForm"
        );


    if (searchForm) {

        searchForm.addEventListener(
            "submit",
            function () {

                const button =
                    searchForm.querySelector(
                        "button[type='submit']"
                    );


                const input =
                    searchForm.querySelector(
                        "input[name='query']"
                    );


                if (
                    input &&
                    input.value.trim() === ""
                ) {

                    return;

                }


                if (button) {

                    button.disabled = true;

                    button.innerHTML =
                        "Searching...";

                }

            }
        );

    }


    /* =====================================================
       SEARCH SUGGESTION CARDS
       ===================================================== */

    const searchInput =
        document.getElementById(
            "searchQuery"
        );


    const suggestionCards =
        document.querySelectorAll(
            ".suggestion-card"
        );


    suggestionCards.forEach(
        function (card) {

            card.addEventListener(
                "click",
                function () {

                    if (
                        searchInput &&
                        searchForm
                    ) {

                        searchInput.value =
                            this.dataset.query;

                        searchForm.requestSubmit();

                    }

                }
            );

        }
    );


    /* =====================================================
       TRY ANOTHER SEARCH
       ===================================================== */

    const clearSearchButton =
        document.getElementById(
            "clearSearch"
        );


    if (clearSearchButton) {

        clearSearchButton.addEventListener(
            "click",
            function () {

                window.location.href =
                    "/search";

            }
        );

    }

});