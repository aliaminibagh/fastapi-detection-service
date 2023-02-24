$(document).ready(function () {

    // create a list of URLs
    var urls = [
        '/api/knife/infer-image',
        '/api/arms/infer-image',
        '/api/fight/infer-image',
        '/api/fire/infer-image',
        '/api/smoke/infer-image',
        '/api/emotions/infer-image',
        '/api/human/infer-image',
        '/api/yolov8/infer-image',
        '/api/face/infer-image',
        '/api/Plate/infer-image'
    ]
    selectedApi = urls[0];

    // $("form#f1").submit(function (e) {
    //     e.preventDefault();
    //     var formData = new FormData(this);

    //     $.ajax({
    //         url: '/api/save_file',
    //         type: 'POST',
    //         data: formData,
    //         success: function (data) {
    //             alert(JSON.stringify(data))
    //         },
    //         fail: function (data) {
    //             console.log(data);
    //         },
    //         cache: false,
    //         contentType: false,
    //         processData: false
    //     });
    // });

    $("#RunButton").click(function () {
        $('#RunButton').toggleClass('is-loading');
        var form = $("form#f1");
        // you can't pass Jquery form it has to be javascript form object
        var formData = new FormData(form[0]);

        // //if ($(form).valid()) {
        //     $.ajax({
        //         type: "POST",
        //         url: '/api/object-to-json',
        //         //dataType: 'json', //not sure but works for me without this
        //         data: formData,
        //         contentType: false, //this is requireded please see answers above
        //         processData: false, //this is requireded please see answers above
        //         //cache: false, //not sure but works for me without this
        //         error   : function (data) {
        //             alert(JSON.stringify(data));
        //             $('#RunButton').toggleClass('is-loading');
        //         },
        //         success : function (data) {
        //             alert(JSON.stringify(data));
        //             $('#RunButton').toggleClass('is-loading');
        //         },
        //     });
        // //}

        $.ajax({
            type: "POST",
            url: selectedApi,
            //dataType: 'json', //not sure but works for me without this
            data: formData,
            contentType: false, //this is requireded please see answers above
            processData: false, //this is requireded please see answers above
            //cache: false, //not sure but works for me without this
            error: function (data) {
                alert('Error');
            },
            success: function (data) {
                // $("#resultbox").attr("src", 'data:image/jpeg;base64,' + data.img_base64);
                console.log(data);
                $("#resultbox").attr("src", data.image);
                $("#jsonbox").val(JSON.stringify(data.result));
                $('#RunButton').toggleClass('is-loading');
            },
        });

    });


    let tabsWithContent = (function () {
        let tabs = document.querySelectorAll('.tabs li');
        let tabsContent = document.querySelectorAll('.tab-content');

        let deactvateAllTabs = function () {
            tabs.forEach(function (tab) {
                tab.classList.remove('is-active');
            });
        };

        let hideTabsContent = function () {
            tabsContent.forEach(function (tabContent) {
                tabContent.classList.remove('is-active');
            });
        };

        let activateTabsContent = function (tab) {
            tabsContent[getIndex(tab)].classList.add('is-active');
            // log the index of the tab
            selectedApi = urls[getIndex(tab)];
        };

        let getIndex = function (el) {
            return [...el.parentElement.children].indexOf(el);
        };

        tabs.forEach(function (tab) {
            tab.addEventListener('click', function () {
                deactvateAllTabs();
                hideTabsContent();
                tab.classList.add('is-active');
                activateTabsContent(tab);
            });
        })

        tabs[0].click();
    })();

});