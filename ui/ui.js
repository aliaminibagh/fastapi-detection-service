$(document).ready(function () {
  $("#file").change(function () {
    var i = $(this).prev("label").clone();
    var file = $("#file")[0].files[0].name;
    $(this).prev("label").text(file);
  });

  var post_mode = "isimage";
  var radios = document.querySelectorAll(
    'input[type=radio][name="fav_language"]'
  );
  function changeHandler(event) {
    post_mode = this.value;
    console.log("post-mode", post_mode);
    if (post_mode === "istext") {
      $("#textRow").show();
      $("#OurApis").hide();
    } else {
      $("#textRow").hide();
      $("#OurApis").show();
    }
  }

  Array.prototype.forEach.call(radios, function (radio) {
    radio.addEventListener("change", changeHandler);
  });

  // create a list of URLs
  var urls = [
    "/api/knife/infer-",
    "/api/arms/infer-",
    "/api/fight/infer-",
    "/api/fire/infer-",
    "/api/smoke/infer-",
    "/api/emotions/infer-",
    "/api/human/infer-",
    "/api/yolov8/infer-",
    "/api/face/infer-",
    "/api/plate/infer-",
  ];
  selectedApi = urls[0];
  $("#post_to_text_api").click(function () {
    var form = $("#text_form").serialize();
    $.ajax({
        type: "POST",
        url: "/api/infer_text?" + form,
        contentType: false, //this is requireded please see answers above
        processData: false, //this is requireded please see answers above
        error: function (data) {
          alert("Error");
          console.log(JSON.stringify(data));
        },
        success: function (data) {
          $("#text_result").val(JSON.stringify(data.result));
        },
      });
  });
  $("#RunButton").click(function () {
    var form = $("form#f1");
    // you can't pass Jquery form it has to be javascript form object
    var formData = new FormData(form[0]);
    $("#RunButton").toggleClass("is-loading");
    $.ajax({
      type: "POST",
      url: selectedApi,
      //dataType: 'json', //not sure but works for me without this
      data: formData,
      contentType: false, //this is requireded please see answers above
      processData: false, //this is requireded please see answers above
      //cache: false, //not sure but works for me without this
      error: function (data) {
        alert("Error");
      },
      success: function (data) {
        // $("#resultbox").attr("src", 'data:image/jpeg;base64,' + data.img_base64);
        console.log(data);
        console.log(JSON.stringify(data));
        // if is_video is true, show the video, else show the image
        if ($('#isvideo').is(':checked')) {
          $('#videobox').show();
          $('#resultbox').hide();
          $('#jsonbox').hide();
          $("#videobox").attr("src", data.video);
          $("#resultbox").attr("src", "");
        } else {
          $('#videobox').hide();
          $('#resultbox').show();
          $('#jsonbox').show();
          $("#videobox").attr("src", "");
          $("#resultbox").attr("src", data.image);
          $("#jsonbox").val(JSON.stringify(data.result));
        }

        // $("#resultbox").attr("src", data.image);
        // $("#jsonbox").val(JSON.stringify(data.result));
        $("#RunButton").toggleClass("is-loading");
      },
    });
  });

  var input = document.getElementById("file");
  var infoArea = document.getElementById("file-upload-filename");

  input.addEventListener("change", showFileName);

  function showFileName(event) {
    // the change event gives us the input it occurred in
    var input = event.srcElement;

    // the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
    var fileName = input.files[0].name;

    // use fileName however fits your app best, i.e. add it into a div
    infoArea.textContent = "اسم فایل : " + fileName;
  }

  let tabsWithContent = (function () {
    let tabs = document.querySelectorAll(".tabs li");
    let tabsContent = document.querySelectorAll(".tab-content");

    let deactvateAllTabs = function () {
      tabs.forEach(function (tab) {
        tab.classList.remove("is-active");
      });
    };

    let hideTabsContent = function () {
      tabsContent.forEach(function (tabContent) {
        tabContent.classList.remove("is-active");
      });
    };

    let activateTabsContent = function (tab) {
      tabsContent[getIndex(tab)].classList.add("is-active");
      // log the index of the tab
      selectedApi = urls[getIndex(tab)];
      // switch based on post_mode
      if (post_mode === "isimage") {
        selectedApi = selectedApi + "image";
      } else if (post_mode === "isvideo") {
        selectedApi = selectedApi + "video";
      }

      // is_video = $('#isvideo').is(':checked');
      // if (is_video) {
      //     selectedApi = selectedApi + 'video';
      // }
      // else {
      //     selectedApi = selectedApi + 'image';
      // }
    };

    let getIndex = function (el) {
      return [...el.parentElement.children].indexOf(el);
    };

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        deactvateAllTabs();
        hideTabsContent();
        tab.classList.add("is-active");
        activateTabsContent(tab);
      });
    });

    tabs[0].click();
  })();
});
