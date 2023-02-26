$(document).ready(function () {
    $('#file').change(function() {
        var i = $(this).prev('label').clone();
        var file = $('#file')[0].files[0].name;
        $(this).prev('label').text(file);
    });


    // create a list of URLs
    var urls = [
        '/api/knife/infer-',
        '/api/arms/infer-',
        '/api/fight/infer-',
        '/api/fire/infer-',
        '/api/smoke/infer-',
        '/api/emotions/infer-',
        '/api/human/infer-',
        '/api/yolov8/infer-',
        '/api/face/infer-',
        '/api/plate/infer-'
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

        // if isvideo radio button is checked, change selectedApi to /api/video, else change it to /api/image
        
        


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
                console.log(JSON.stringify(data));
                // if is_video is true, show the video, else show the image
                if (is_video) {
                    $("#videobox").attr("src", data.video);
                    $("#resultbox").attr("src", "");
                }
                else {
                    $("#videobox").attr("src", "");
                    $("#resultbox").attr("src", data.image);
                    $("#jsonbox").val(JSON.stringify(data.result));
                }

                // $("#resultbox").attr("src", data.image);
                // $("#jsonbox").val(JSON.stringify(data.result));
                $('#RunButton').toggleClass('is-loading');
            },
        });

    });

    var input = document.getElementById('file');
    var infoArea = document.getElementById('file-upload-filename');

    input.addEventListener('change', showFileName);

    function showFileName(event) {

        // the change event gives us the input it occurred in 
        var input = event.srcElement;

        // the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
        var fileName = input.files[0].name;

        // use fileName however fits your app best, i.e. add it into a div
        infoArea.textContent = 'اسم فایل : ' + fileName;
    }

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
            is_video = $('#isvideo').is(':checked');
            if (is_video) {
                selectedApi = selectedApi + 'video';
            }
            else {
                selectedApi = selectedApi + 'image';
            }
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

