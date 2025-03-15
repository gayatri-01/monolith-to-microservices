/**
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.apache.geronimo.daytrader.javaee6.web;

import java.io.IOException;
import java.net.URLEncoder;

import javax.servlet.ServletConfig;
import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.geronimo.daytrader.javaee6.core.direct.TradeJDBCDirect;
import org.apache.geronimo.daytrader.javaee6.core.direct.TradeJEEDirect;
import org.apache.geronimo.daytrader.javaee6.core.direct.TradeJPADirect;
import org.apache.geronimo.daytrader.javaee6.utils.Log;
import org.apache.geronimo.daytrader.javaee6.utils.TradeConfig;
import org.json.JSONObject;


/**
 * 
 * TradeAppServlet provides the standard web interface to Trade and can be
 * accessed with the Go Trade! link. Driving benchmark load using this interface
 * requires a sophisticated web load generator that is capable of filling HTML
 * forms and posting dynamic data.
 */

@WebServlet("/app")
public class TradeAppServlet extends HttpServlet {


	/*private final static String CLIENT_ID = "352093438450-1ljnv0344ghiilr3j9hubl496jecdgde.apps.googleusercontent.com";
	private final static String CLIENT_SECRET ="8eqSmpK72T4azYaWQhxcmZgD";  */

	private final static String CLIENT_ID = "352093438450-ofdjons8g7tnk9ur19jok2hbmqbq4hfr.apps.googleusercontent.com";
	private final static String CLIENT_SECRET ="HgHYt7d-L2ur6ySTHozdW2Xr";
	private final static String GOOGLE_SSO_URL="https://www.googleapis.com/oauth2/v1/userinfo?access_token=";
	private final static String HOST_URL="http://ec2-54-167-178-47.compute-1.amazonaws.com:8080/";

	/**
	 * Servlet initialization method.
	 */
	public void init(ServletConfig config) throws ServletException {
		super.init(config);
		java.util.Enumeration en = config.getInitParameterNames();
		while (en.hasMoreElements()) {
			String parm = (String) en.nextElement();
			String value = config.getInitParameter(parm);
			TradeConfig.setConfigParam(parm, value);
		}
		try {
			if (TradeConfig.runTimeMode == TradeConfig.JDBC) {
				TradeJDBCDirect.init();
			} else if (TradeConfig.runTimeMode == TradeConfig.JPA) {
				TradeJPADirect.init();
			} else {
				TradeJEEDirect.init();
			}
		} catch (Exception e) {
			Log.error(e, "TradeAppServlet:init -- Error initializing TradeDirect");
		}
	}

	/**
	 * Returns a string that contains information about TradeScenarioServlet
	 * 
	 * @return The servlet information
	 */
	public java.lang.String getServletInfo() {
		return "TradeAppServlet provides the standard web interface to Trade";
	}

	/**
	 * Process incoming HTTP GET requests
	 * 
	 * @param request
	 *            Object that encapsulates the request to the servlet
	 * @param response
	 *            Object that encapsulates the response from the servlet
	 */
	public void doGet(javax.servlet.http.HttpServletRequest request,
			javax.servlet.http.HttpServletResponse response)
					throws ServletException, IOException {
		performTask(request, response);
	}

	/**
	 * Process incoming HTTP POST requests
	 * 
	 * @param request
	 *            Object that encapsulates the request to the servlet
	 * @param response
	 *            Object that encapsulates the response from the servlet
	 */
	public void doPost(javax.servlet.http.HttpServletRequest request,
			javax.servlet.http.HttpServletResponse response)
					throws ServletException, IOException {
		performTask(request, response);
	}

	/**
	 * Main service method for TradeAppServlet
	 * 
	 * @param request
	 *            Object that encapsulates the request to the servlet
	 * @param response
	 *            Object that encapsulates the response from the servlet
	 */
	@SuppressWarnings("deprecation")
	public void performTask(HttpServletRequest req, HttpServletResponse resp)
			throws ServletException, IOException {

		String action = null;
		String userID = null;
		// String to create full dispatch path to TradeAppServlet w/ request
		// Parameters
		String dispPath = null; // Dispatch Path to TradeAppServlet

		resp.setContentType("text/html");
		TradeServletAction tsAction = new TradeServletAction();

		// Dyna - need status string - prepended to output
		action = req.getParameter("action");

		ServletContext ctx = getServletConfig().getServletContext();

		if (action == null) {
			tsAction.doWelcome(ctx, req, resp, "");
			return;
		} else if (action.equals("login")) {
			userID = req.getParameter("uid");
			String passwd = req.getParameter("passwd");
			String inScenario = req.getParameter("inScenario");
			try {
				tsAction.doLogin(ctx, req, resp, userID, passwd);
			} catch (ServletException se) {
				tsAction.doWelcome(ctx, req, resp, se.getMessage());
			}
			return;
		} else if (action.equals("register")) {
			userID = req.getParameter("user id");
			String passwd = req.getParameter("passwd");
			String cpasswd = req.getParameter("confirm passwd");
			String fullname = req.getParameter("Full Name");
			String ccn = req.getParameter("Credit Card Number");
			String money = req.getParameter("money");
			String email = req.getParameter("email");
			String smail = req.getParameter("snail mail");
			tsAction.doRegister(ctx, req, resp, userID, passwd, cpasswd,
					fullname, ccn, money, email, smail);
			return;
		} else if (action.equals("registerproxy")) {
			String code =  req.getParameter("code");
			String state = req.getParameter("state");
			String foros = "code="+code +
					"&client_id=" +CLIENT_ID + 
//					"&client_secret=" +CLIENT_SECRET + "&grant_type=authorization_code&redirect_uri=http://localhost:8080/daytrader/app?action=registerproxy";
					"&client_secret=" +CLIENT_SECRET + "&grant_type=authorization_code&redirect_uri="+HOST_URL+"daytrader/app?action=registerproxy";
			

			String url = "https://accounts.google.com/o/oauth2/token";
			System.out.println("Url:" + url);
			HttpClient client = new HttpClient();
			PostMethod httppost = new PostMethod(url);
			//httppost
			httppost.addRequestHeader("Content-Type", "application/x-www-form-urlencoded");

			//ByteArrayRequestEntity entity = new ByteArrayRequestEntity(foros.getBytes("UTF-8"));
			//httppost.set
			System.out.println("Foros :" + foros);
			httppost.addParameter("grant_type", "authorization_code");
			httppost.addParameter("code", code);
			httppost.addParameter("client_id", CLIENT_ID);
			httppost.addParameter("client_secret", CLIENT_SECRET);

			//httppost.addParameter("redirect_uri","http://localhost:8080/daytrader/app?action=registerproxy" );		

			// httppost.setRequestBody(foros);
			// httppost.setEntity(entity);

			httppost.addParameter("redirect_uri",HOST_URL+"daytrader/app?action=registerproxy" );		
			
	       // httppost.setRequestBody(foros);
	       // httppost.setEntity(entity);

			String accessToken = null;
			try {
				// client.set

				int responseCode =  client.executeMethod(httppost);
				String jsonResponse = httppost.getResponseBodyAsString();

				System.out.println("jsonResponse:" + jsonResponse);
				JSONObject jsonParser = new JSONObject(jsonResponse);
				accessToken = (String)jsonParser.getString("access_token");
				System.out.println("Access Token:" + accessToken);

				GetMethod userInfoGet = new GetMethod(GOOGLE_SSO_URL+accessToken);

				String googleId = null;
				String email = null;
				String name = null;
				String firstName = null;
				String lastName = null;
				try {
					responseCode = client.executeMethod(userInfoGet);
					String userInfo =userInfoGet.getResponseBodyAsString();

					JSONObject userInfoParser = new JSONObject(userInfo);

					googleId = (String) userInfoParser.getString("id");
					email = (String) userInfoParser.getString("email");
					name = (String) userInfoParser.getString("name");
					firstName = (String) userInfoParser.getString("given_name");
					lastName = (String) userInfoParser.getString("family_name");
					userID = email;
					try {//tsAction.
						String results = "";
						try {
							tsAction.doCheck(ctx, req, resp, userID,results );
							tsAction.doLogin(ctx, req, resp, userID);
						}catch (Exception ex) {
							results = "NOT_FOUND";
							tsAction.doRegister(ctx, req, resp, email, "password", "password",
									firstName + " " + lastName,  "CCN", "500", email, "Please update Address");
							req.getRequestDispatcher("/app?action=login&uid="+email+"&passwd=password").forward(req, resp); 
						}
						/*Log.error("Looking for request status: "+ results);
						if (results.equals("FOUND"))
							
						else */

					} catch (Exception ex) {
						//tsAction.doLogin(ctx, req, resp, userID, passwd);
					}
					

				} catch (IOException e) {
					throw new RuntimeException(e);
				} 


			} catch (Exception e) {
				throw new RuntimeException(e);
			}

		}

		// The rest of the operations require the user to be logged in -
		// Get the Session and validate the user.
		HttpSession session = req.getSession();
		userID = (String) session.getAttribute("uidBean");

		if (userID == null) {
			System.out
			.println("TradeAppServlet service error: User Not Logged in");
			tsAction.doWelcome(ctx, req, resp, "User Not Logged in");
			return;
		}
		if (action.equals("quotes")) {
			String symbols = req.getParameter("symbols");
			tsAction.doQuotes(ctx, req, resp, userID, symbols);
		} else if (action.equals("buy")) {
			String symbol = req.getParameter("symbol");
			String quantity = req.getParameter("quantity");
			tsAction.doBuy(ctx, req, resp, userID, symbol, quantity);
		} else if (action.equals("sell")) {
			int holdingID = Integer.parseInt(req.getParameter("holdingID"));
			tsAction.doSell(ctx, req, resp, userID, new Integer(holdingID));
		} else if (action.equals("portfolio")
				|| action.equals("portfolioNoEdge")) {
			tsAction.doPortfolio(ctx, req, resp, userID, "Portfolio as of "
					+ new java.util.Date());
		} else if (action.equals("logout")) {
			tsAction.doLogout(ctx, req, resp, userID);
		} else if (action.equals("home")) {
			tsAction.doHome(ctx, req, resp, userID, "Ready to Trade");
		} else if (action.equals("account")) {
			tsAction.doAccount(ctx, req, resp, userID, "");
		} else if (action.equals("update_profile")) {
			String password = req.getParameter("password");
			String cpassword = req.getParameter("cpassword");
			String fullName = req.getParameter("fullname");
			String address = req.getParameter("address");
			String creditcard = req.getParameter("creditcard");
			String email = req.getParameter("email");
			tsAction.doAccountUpdate(ctx, req, resp, userID,
					password == null ? "" : password.trim(),
							cpassword == null ? "" : cpassword.trim(),
									fullName == null ? "" : fullName.trim(),
											address == null ? "" : address.trim(),
													creditcard == null ? "" : creditcard.trim(),
															email == null ? "" : email.trim());
		} else {
			System.out.println("TradeAppServlet: Invalid Action=" + action);
			tsAction.doWelcome(ctx, req, resp,
					"TradeAppServlet: Invalid Action" + action);
		}
	}

	private void sendRedirect(HttpServletResponse resp, String page)
			throws ServletException, IOException {
		resp.sendRedirect(resp.encodeRedirectURL(page));
	}

	// URL Path Prefix for dispatching to TradeAppServlet
	private final static String tasPathPrefix = "/app?action=";

}
