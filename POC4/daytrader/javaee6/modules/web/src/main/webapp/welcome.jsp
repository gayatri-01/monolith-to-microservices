<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<HTML>
<HEAD>
<TITLE>Trade Login</TITLE>
<LINK rel="stylesheet" href="style.css" type="text/css" />
</HEAD>
<BODY bgcolor="#ffffff" link="#000099">
<%@ page session="false"%>
<TABLE style="font-size: smaller">
    <TBODY>
        <TR>
            <TD bgcolor="#c93333" align="left" width="640" height="10"><B><FONT
                color="#ffffff">DayTrader Login</FONT></B></TD>
            <TD align="center" bgcolor="#000000" width="100" height="10"><FONT
                color="#ffffff"><B>DayTrader</B></FONT></TD>
        </TR>
    </TBODY>
</TABLE>
<TABLE width="100%" height="30">
    <TBODY>
        <TR>
            <TD></TD>
            <TD><FONT color="#ff0033"><FONT color="#ff0033"><FONT color="#ff0033"><% String results;
results = (String) request.getAttribute("results");
if ( results != null )out.print(results);
%></FONT></FONT></FONT></TD>
            <TD></TD>
        </TR>
    </TBODY>
</TABLE>
<DIV align="left"></DIV>
<TABLE width="616">
    <TBODY>
        <TR>
            <TD width="2%" bgcolor="#e7e4e7" rowspan="4"></TD>
            <TD width="98%"><B>Log in</B>
            <HR>
            </TD>
        </TR>
        <TR>
            <TD align="right"><FONT size="-1">Username &nbsp; &nbsp; &nbsp;&nbsp;
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Password &nbsp; &nbsp; &nbsp;
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</FONT></TD>
        </TR>
        <TR>
            <TD align="right">
            <FORM action="app" method="POST"><INPUT size="10" type="text"
                name="uid" value="uid:0"> &nbsp; &nbsp; &nbsp; &nbsp; <INPUT
                size="10" type="password" name="passwd" value="xxx"> &nbsp; <INPUT
                type="submit" value="Log in"><INPUT type="hidden" name="action"
                value="login"></FORM>
            </TD>
        </TR>
        <TR>
           <TD align="right">
           <a href="javascript:top.location.href='https://accounts.google.com/o/oauth2/auth?scope=https://www.googleapis.com/auth/userinfo.profile+https://www.googleapis.com/auth/userinfo.email&state=%2Fprofile&response_type=code&client_id=352093438450-ofdjons8g7tnk9ur19jok2hbmqbq4hfr.apps.googleusercontent.com&redirect_uri=http://ec2-54-167-178-47.compute-1.amazonaws.com:8080/daytrader/app?action=registerproxy';"> Login using Google SSO</a>
           </TD>
        </TR>
    </TBODY>
</TABLE>
<TABLE width="616">
    <TBODY>
        <TR>
            <TD width="2%"></TD>
            <TD width="98%">
            <HR>
            </TD>
        </TR>
        <TR>
            <TD bgcolor="#e7e4e7" rowspan="4"></TD>
            <TD><B><FONT size="-1" color="#000000">First time user? &nbsp;Please
            Register</FONT></B></TD>
        </TR>
        <TR>
            <TD></TD>
        </TR>
        <TR>
            <TD align="right">
            <BLOCKQUOTE><A href="register.jsp">Register&nbsp;With&nbsp;DayTrader</A>
            </BLOCKQUOTE>
            </TD>
        </TR>
        <TR>
            <TD>
            <HR>
            </TD>
        </TR>
    </TBODY>
</TABLE>
<TABLE height="54" style="font-size: smaller">
    <TBODY>
        <TR>
            <TD colspan="2">
            <HR>
            </TD>
        </TR>
        <TR>
            <TD colspan="2"></TD>
        </TR>
        <TR>
            <TD bgcolor="#c93333" align="left" width="640" height="10"><B><FONT
                color="#ffffff">DayTrader Login</FONT></B></TD>
            <TD align="center" bgcolor="#000000" width="100" height="10"><FONT
                color="#ffffff"><B>DayTrader</B></FONT></TD>
        </TR>
    </TBODY>
</TABLE>
</BODY>
</HTML>
