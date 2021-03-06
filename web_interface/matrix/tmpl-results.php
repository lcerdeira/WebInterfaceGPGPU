<!DOCTYPE html>
<html lang="en">
<head>
<title>Simple Table</title>
<?php include_once('../includes/func.php'); ?>
<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.4/jquery.min.js"></script>
<script type="text/javascript">
 $(function() {
		/* For zebra striping */
        $("table tr:nth-child(odd)").addClass("odd-row");
		/* For cell text alignment */
		$("table td:first-child, table th:first-child").addClass("first");
		/* For removing the last border */
		$("table td:last-child, table th:last-child").addClass("last");
});
</script>

<link rel="stylesheet" href="../css/style-result.css">

</head>
<body>

<div id="content">
<?php formatArrayToTable($_POST); ?>
</div>

</body>
</html>
