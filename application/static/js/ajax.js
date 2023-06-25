$(document).ready(function () {
  $('input[type="radio"]').click(function () {
    console.log("Button clicked");
    var movieId = $(this).attr("name");
    var voteValue = $(this).val() == "like" ? 5 : 0;

    var data = { movie_id: movieId, vote_value: voteValue };

    $.ajax({
      type: "POST",
      url: "/vote",
      data: JSON.stringify(data),
      contentType: "application/json",
      success: function (response) {
        console.log("Vote successfully recorded.");
      },
      error: function (error) {
        console.log("Error recording vote: " + error);
      },
    });
  });
});
